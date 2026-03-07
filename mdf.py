import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load test data into dataframe
df = pd.read_csv("test.csv", header=0)

df.columns = ["ch_1_time", "ch_1_voltage", 
              "ch_2_time", "ch_2_voltage", 
              "ch_3_time", "ch_3_voltage", 
              "ch_4_time", "ch_4_voltage"]

# pick one channel
time = df["ch_1_time"].to_numpy()
voltage = df["ch_1_voltage"].to_numpy()

# sampling interval and sampling frequency
dt = np.mean(np.diff(time))   # seconds
Fs = 1 / dt                   # Hz

# number of samples
N = len(voltage)

# remove DC offset
voltage = voltage - np.mean(voltage)

# FFT
fft_vals = np.fft.fft(voltage)
freqs = np.fft.fftfreq(N, d=dt)

# keep only positive frequencies
positive = freqs >= 0
freqs = freqs[positive]
amplitude = np.abs(fft_vals[positive]) / N

# optional: double amplitudes except DC term for single-sided spectrum
amplitude[1:-1] *= 2

# plot
plt.plot(freqs, amplitude)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.title("FFT of EMG Channel 1")
plt.xlim(0, 500)  # EMG usually lives in this range
plt.grid(True)
plt.show()
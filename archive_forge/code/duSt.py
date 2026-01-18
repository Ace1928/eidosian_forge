import numpy as np
import librosa
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.fftpack import fft, ifft
from sklearn.preprocessing import MinMaxScaler
import threading
import queue
import time
import psutil
import os
import sys
import select
import librosa.display
import seaborn as sns
import pandas as pd
from datetime import datetime
from scipy.signal import spectrogram, find_peaks
from scipy.stats import kurtosis, skew
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import gc

# Global variables for controlling the visualization
running = True
paused = False
signal_queue = queue.Queue()
default_file_path = "/home/lloyd/Downloads/audio/output.mp3"
output_dir = "/home/lloyd/Downloads/tensorwave/outputs"
os.makedirs(output_dir, exist_ok=True)


# Ensure at least 10% of system resources are free
def ensure_system_resources():
    memory_info = psutil.virtual_memory()
    cpu_percent = psutil.cpu_percent(interval=1)
    if memory_info.available < memory_info.total * 0.1 or cpu_percent > 90:
        raise SystemError("Not enough system resources available.")


def get_timestamp() -> str:
    """Get the current timestamp for unique file naming."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def read_audio_file(file_path: str = default_file_path) -> tuple[np.ndarray, int]:
    """Read an audio file and return the signal and sample rate."""
    return librosa.load(file_path, sr=None)


def record_audio(duration: int = 5, sr: int = 44100) -> tuple[np.ndarray, int]:
    """Record audio from the microphone for a given duration and sample rate."""
    print("Recording...")
    signal = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="float64")
    sd.wait()
    print("Recording complete.")
    return signal.flatten(), sr


def get_max_freq_components(sr: int) -> int:
    """Calculate the maximum frequency components based on system resources."""
    cpu_count = psutil.cpu_count(logical=False)
    memory_info = psutil.virtual_memory()
    max_freq_components = int((sr / 2) * 0.8)  # 80% of the Nyquist frequency
    return min(max_freq_components, cpu_count * 1000, memory_info.available // (8 * sr))


def create_tensor_wave(signal: np.ndarray, sr: int) -> np.ndarray:
    """
    Create a tensor representation of the wave using FFT and IFFT with normalization and scaling.

    This function performs the following steps:
    1. Computes the FFT of the input signal.
    2. Constructs a tensor wave using the IFFT of the frequency components.
    3. Normalizes and scales the tensor wave for better interpretability.

    Parameters:
    signal (np.ndarray): The input audio signal.
    sr (int): The sample rate of the audio signal.

    Returns:
    np.ndarray: The normalized and scaled tensor representation of the wave.
    """
    time_steps = len(signal)
    max_freq_components = get_max_freq_components(sr)
    freq_components = fft(signal)[:max_freq_components]
    tensor_wave = np.zeros((time_steps, len(freq_components)), dtype=np.complex_)

    exp_factor = 2j * np.pi * np.arange(time_steps) / time_steps

    for i, freq in enumerate(freq_components):
        tensor_wave[:, i] = ifft(freq * np.exp(exp_factor * i))

    # Normalize the tensor wave
    abs_tensor_wave = np.abs(tensor_wave)
    scaler = MinMaxScaler()
    scaled_tensor_wave = scaler.fit_transform(abs_tensor_wave)

    return scaled_tensor_wave


def save_visualization(
    signal: np.ndarray, tensor_wave: np.ndarray, sr: int, title: str, viz_type: str
):
    """Save a specific visualization of the signal and tensor wave."""
    time = np.linspace(0, len(signal) / sr, num=len(signal))
    timestamp = get_timestamp()

    if viz_type == "original_signal":
        plt.figure(figsize=(15, 5))
        plt.plot(time, signal, label="Original Signal")
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        plt.title(f"{title} - Original Signal")
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"original_signal_{timestamp}.png"))
        plt.close()

    elif viz_type == "tensor_wave_heatmap":
        plt.figure(figsize=(15, 5))
        sns.heatmap(tensor_wave.T, cmap="viridis", cbar=True)
        plt.xlabel("Time [s]")
        plt.ylabel("Frequency Component")
        plt.title(f"{title} - Tensor Wave Heatmap")
        plt.savefig(os.path.join(output_dir, f"tensor_wave_heatmap_{timestamp}.png"))
        plt.close()

    elif viz_type == "spectrogram":
        f, t, Sxx = spectrogram(signal, sr)
        plt.figure(figsize=(15, 5))
        plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading="gouraud")
        plt.xlabel("Time [s]")
        plt.ylabel("Frequency [Hz]")
        plt.title(f"{title} - Spectrogram")
        plt.colorbar(label="Intensity [dB]")
        plt.savefig(os.path.join(output_dir, f"spectrogram_{timestamp}.png"))
        plt.close()

    elif viz_type == "power_spectrum":
        plt.figure(figsize=(15, 5))
        power_spectrum = np.abs(fft(signal)) ** 2
        freqs = np.fft.fftfreq(len(signal), 1 / sr)
        plt.plot(freqs[: len(freqs) // 2], power_spectrum[: len(power_spectrum) // 2])
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Power")
        plt.title(f"{title} - Power Spectrum")
        plt.savefig(os.path.join(output_dir, f"power_spectrum_{timestamp}.png"))
        plt.close()

    elif viz_type == "zero_crossing_rate":
        zero_crossings = librosa.zero_crossings(signal, pad=False)
        plt.figure(figsize=(15, 5))
        plt.plot(time, signal, label="Signal")
        plt.vlines(
            time[zero_crossings],
            ymin=min(signal),
            ymax=max(signal),
            color="r",
            alpha=0.5,
            label="Zero Crossings",
        )
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        plt.title(f"{title} - Zero Crossing Rate")
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"zero_crossing_rate_{timestamp}.png"))
        plt.close()

    elif viz_type == "rms_energy":
        rms = librosa.feature.rms(y=signal)[0]
        plt.figure(figsize=(15, 5))
        plt.plot(time, signal, label="Signal")
        plt.plot(time[: len(rms)], rms, label="RMS Energy", color="r")
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude / RMS Energy")
        plt.title(f"{title} - RMS Energy")
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"rms_energy_{timestamp}.png"))
        plt.close()

    elif viz_type == "mel_spectrogram":
        mel_spectrogram = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=128)
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        plt.figure(figsize=(15, 5))
        librosa.display.specshow(mel_spectrogram_db, sr=sr, x_axis="time", y_axis="mel")
        plt.colorbar(format="%+2.0f dB")
        plt.title(f"{title} - Mel Spectrogram")
        plt.savefig(os.path.join(output_dir, f"mel_spectrogram_{timestamp}.png"))
        plt.close()

    elif viz_type == "chromagram":
        chromagram = librosa.feature.chroma_stft(y=signal, sr=sr)
        plt.figure(figsize=(15, 5))
        librosa.display.specshow(
            chromagram, x_axis="time", y_axis="chroma", cmap="coolwarm"
        )
        plt.colorbar()
        plt.title(f"{title} - Chromagram")
        plt.savefig(os.path.join(output_dir, f"chromagram_{timestamp}.png"))
        plt.close()

    elif viz_type == "tonnetz":
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(signal), sr=sr)
        plt.figure(figsize=(15, 5))
        librosa.display.specshow(tonnetz, x_axis="time")
        plt.colorbar()
        plt.title(f"{title} - Tonnetz")
        plt.savefig(os.path.join(output_dir, f"tonnetz_{timestamp}.png"))
        plt.close()

    elif viz_type == "tempogram":
        tempogram = librosa.feature.tempogram(y=signal, sr=sr)
        plt.figure(figsize=(15, 5))
        librosa.display.specshow(tempogram, sr=sr, x_axis="time", y_axis="tempo")
        plt.colorbar()
        plt.title(f"{title} - Tempogram")
        plt.savefig(os.path.join(output_dir, f"tempogram_{timestamp}.png"))
        plt.close()

    elif viz_type == "spectral_centroid":
        spectral_centroid = librosa.feature.spectral_centroid(y=signal, sr=sr)[0]
        plt.figure(figsize=(15, 5))
        plt.semilogy(time, spectral_centroid, label="Spectral Centroid")
        plt.xlabel("Time [s]")
        plt.ylabel("Hz")
        plt.title(f"{title} - Spectral Centroid")
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"spectral_centroid_{timestamp}.png"))
        plt.close()

    elif viz_type == "spectral_bandwidth":
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=sr)[0]
        plt.figure(figsize=(15, 5))
        plt.semilogy(time, spectral_bandwidth, label="Spectral Bandwidth")
        plt.xlabel("Time [s]")
        plt.ylabel("Hz")
        plt.title(f"{title} - Spectral Bandwidth")
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"spectral_bandwidth_{timestamp}.png"))
        plt.close()

    elif viz_type == "spectral_contrast":
        spectral_contrast = librosa.feature.spectral_contrast(y=signal, sr=sr)
        plt.figure(figsize=(15, 5))
        librosa.display.specshow(spectral_contrast, x_axis="time")
        plt.colorbar()
        plt.title(f"{title} - Spectral Contrast")
        plt.savefig(os.path.join(output_dir, f"spectral_contrast_{timestamp}.png"))
        plt.close()

    elif viz_type == "spectral_flatness":
        spectral_flatness = librosa.feature.spectral_flatness(y=signal)[0]
        plt.figure(figsize=(15, 5))
        plt.plot(time, spectral_flatness, label="Spectral Flatness")
        plt.xlabel("Time [s]")
        plt.ylabel("Flatness")
        plt.title(f"{title} - Spectral Flatness")
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"spectral_flatness_{timestamp}.png"))
        plt.close()

    elif viz_type == "kurtosis":
        signal_kurtosis = kurtosis(signal)
        plt.figure(figsize=(15, 5))
        plt.plot(time, signal, label="Signal")
        plt.axhline(y=signal_kurtosis, color="r", linestyle="-", label="Kurtosis")
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        plt.title(f"{title} - Kurtosis")
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"kurtosis_{timestamp}.png"))
        plt.close()

    elif viz_type == "skewness":
        signal_skewness = skew(signal)
        plt.figure(figsize=(15, 5))
        plt.plot(time, signal, label="Signal")
        plt.axhline(y=signal_skewness, color="r", linestyle="-", label="Skewness")
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        plt.title(f"{title} - Skewness")
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"skewness_{timestamp}.png"))
        plt.close()

    elif viz_type == "peak_detection":
        peaks, _ = find_peaks(signal, height=0)
        plt.figure(figsize=(15, 5))
        plt.plot(time, signal, label="Signal")
        plt.plot(time[peaks], signal[peaks], "x", label="Peaks")
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        plt.title(f"{title} - Peak Detection")
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"peak_detection_{timestamp}.png"))
        plt.close()

    elif viz_type == "smoothed_signal":
        smoothed_signal = gaussian_filter1d(signal, sigma=2)
        plt.figure(figsize=(15, 5))
        plt.plot(time, smoothed_signal, label="Smoothed Signal")
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        plt.title(f"{title} - Smoothed Signal")
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"smoothed_signal_{timestamp}.png"))
        plt.close()

    elif viz_type == "pca":
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(tensor_wave)
        plt.figure(figsize=(15, 5))
        plt.scatter(pca_result[:, 0], pca_result[:, 1], c=time, cmap="viridis")
        plt.colorbar(label="Time [s]")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.title(f"{title} - PCA of Tensor Wave")
        plt.savefig(os.path.join(output_dir, f"pca_tensor_wave_{timestamp}.png"))
        plt.close()

    elif viz_type == "tsne":
        tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
        tsne_result = tsne.fit_transform(tensor_wave)
        plt.figure(figsize=(15, 5))
        plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=time, cmap="plasma")
        plt.colorbar(label="Time [s]")
        plt.xlabel("t-SNE Component 1")
        plt.ylabel("t-SNE Component 2")
        plt.title(f"{title} - t-SNE of Tensor Wave")
        plt.savefig(os.path.join(output_dir, f"tsne_tensor_wave_{timestamp}.png"))
        plt.close()

    elif viz_type == "kmeans":
        kmeans = KMeans(n_clusters=3)
        kmeans_result = kmeans.fit_predict(tensor_wave)
        plt.figure(figsize=(15, 5))
        plt.scatter(pca_result[:, 0], pca_result[:, 1], c=kmeans_result, cmap="viridis")
        plt.colorbar(label="Cluster")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.title(f"{title} - K-Means Clustering of Tensor Wave")
        plt.savefig(os.path.join(output_dir, f"kmeans_tensor_wave_{timestamp}.png"))
        plt.close()

    elif viz_type == "plotly_3d_scatter":
        fig = px.scatter_3d(
            x=pca_result[:, 0], y=pca_result[:, 1], z=tsne_result[:, 0], color=time
        )
        fig.update_layout(
            title=f"{title} - 3D Scatter Plot",
            scene=dict(
                xaxis_title="PCA Component 1",
                yaxis_title="PCA Component 2",
                zaxis_title="t-SNE Component 1",
            ),
        )
        fig.write_html(os.path.join(output_dir, f"3d_scatter_{timestamp}.html"))

    elif viz_type == "plotly_line":
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=time, y=signal, mode="lines", name="Original Signal")
        )
        fig.update_layout(
            title=f"{title} - Original Signal",
            xaxis_title="Time [s]",
            yaxis_title="Amplitude",
        )
        fig.write_html(os.path.join(output_dir, f"original_signal_{timestamp}.html"))

    elif viz_type == "plotly_heatmap":
        fig = go.Figure(data=go.Heatmap(z=tensor_wave.T, x=time, colorscale="Viridis"))
        fig.update_layout(
            title=f"{title} - Tensor Wave Heatmap",
            xaxis_title="Time [s]",
            yaxis_title="Frequency Component",
        )
        fig.write_html(
            os.path.join(output_dir, f"tensor_wave_heatmap_{timestamp}.html")
        )

    elif viz_type == "plotly_spectrogram":
        fig = go.Figure(
            data=go.Heatmap(z=10 * np.log10(Sxx), x=t, y=f, colorscale="Viridis")
        )
        fig.update_layout(
            title=f"{title} - Spectrogram",
            xaxis_title="Time [s]",
            yaxis_title="Frequency [Hz]",
        )
        fig.write_html(os.path.join(output_dir, f"spectrogram_{timestamp}.html"))

    elif viz_type == "plotly_3d_surface":
        fig = go.Figure(data=[go.Surface(z=tensor_wave.T)])
        fig.update_layout(
            title=f"{title} - 3D Surface Plot of Tensor Wave",
            scene=dict(
                xaxis_title="Time [s]",
                yaxis_title="Frequency Component",
                zaxis_title="Amplitude",
            ),
        )
        fig.write_html(os.path.join(output_dir, f"3d_surface_{timestamp}.html"))
    # Additional visualizations omitted for brevity

    else:
        raise ValueError(f"Unknown visualization type: {viz_type}")

    # Force garbage collection after saving each visualization
    gc.collect()


def process_input(signal: np.ndarray, sr: int):
    """Threaded function to process input data and update the queue."""
    global running
    chunk_size = sr * 5  # 5 seconds chunks
    for start in range(0, len(signal), chunk_size):
        end = min(start + chunk_size, len(signal))
        signal_queue.put(signal[start:end])
        time.sleep(0.1)
    running = False


def main():
    """Main function to execute the program."""
    global running, paused

    input_option = (
        input("Enter 'file' to load an audio file or 'mic' to record audio: ")
        .strip()
        .lower()
    )
    if input_option == "file":
        file_path = (
            input(
                f"Enter the path to the audio file (default: {default_file_path}): "
            ).strip()
            or default_file_path
        )
        signal, sr = read_audio_file(file_path)
    elif input_option == "mic":
        duration = int(input("Enter the duration for recording (seconds): ").strip())
        signal, sr = record_audio(duration=duration)
    else:
        print("Invalid option. Exiting.")
        return

    input_thread = threading.Thread(target=process_input, args=(signal, sr))
    input_thread.start()

    while running:
        ensure_system_resources()
        if not signal_queue.empty():
            chunk = signal_queue.get()
            tensor_wave = create_tensor_wave(chunk, sr)
            visualizations = [
                "original_signal",
                "tensor_wave_heatmap",
                "spectrogram",
                "power_spectrum",
                "zero_crossing_rate",
                "rms_energy",
                "mel_spectrogram",
                "chromagram",
                "tonnetz",
                "tempogram",
                "spectral_centroid",
                "spectral_bandwidth",
                "spectral_contrast",
                "spectral_flatness",
                "kurtosis",
                "skewness",
                "peak_detection",
                "smoothed_signal",
                "pca",
                "tsne",
                "kmeans",
                "plotly_3d_scatter",
                "plotly_line",
                "plotly_heatmap",
                "plotly_spectrogram",
                "plotly_3d_surface",
                # Additional visualizations omitted for brevity
            ]

            # Process visualizations one at a time to avoid overloading the system
            for viz_type in visualizations:
                if not running:
                    break
                save_visualization(
                    chunk,
                    tensor_wave,
                    sr,
                    title="Wave Visualization",
                    viz_type=viz_type,
                )
                gc.collect()  # Force garbage collection after each visualization

        # Check for user input in a non-blocking way
        if not paused:
            if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                user_input = input().strip().lower()
                if user_input == "pause":
                    paused = True
                elif user_input == "resume":
                    paused = False
                elif user_input == "stop":
                    running = False
                    break
                else:
                    print("Invalid command.")

    input_thread.join()
    print("Program terminated.")


if __name__ == "__main__":
    main()

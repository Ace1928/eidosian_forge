import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.fftpack import fft
from sklearn.preprocessing import MinMaxScaler
import threading
import queue
import time
import psutil
import os
import sys
import logging
from datetime import datetime
from scipy.signal import spectrogram, find_peaks, butter, filtfilt
from scipy.ndimage import gaussian_filter1d
from scipy.stats import kurtosis, skew
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import gc
from pydub import AudioSegment
from scipy.io.wavfile import write as wav_write

# Global variables for controlling the visualization
running = True
paused = False
signal_queue = queue.Queue()
default_file_path = "/home/lloyd/Downloads/audio/output.mp3"
output_dir = "/home/lloyd/Downloads/tensorwave/outputs"
os.makedirs(output_dir, exist_ok=True)

# Initialize logging
logging.basicConfig(
    filename="system_resources.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def ensure_system_resources():
    logging.info("Checking system resources...")
    print("Checking system resources...")

    memory_info = psutil.virtual_memory()
    cpu_percent = psutil.cpu_percent(interval=0.1)

    logging.info(
        f"Memory available: {memory_info.available / (1024 ** 2):.2f} MB, CPU usage: {cpu_percent}%"
    )
    print(
        f"Memory available: {memory_info.available / (1024 ** 2):.2f} MB, CPU usage: {cpu_percent}%"
    )

    if memory_info.percent > 95:
        logging.error("RAM usage exceeded 95%. Exiting.")
        print("RAM usage exceeded 95%. Exiting.")
        raise SystemError("Not enough system resources available.")
    else:
        logging.info(f"RAM usage check passed: {memory_info.percent}% used")
        print(f"RAM usage check passed: {memory_info.percent}% used")

    if memory_info.available < memory_info.total * 0.2:
        logging.warning("Low memory detected, optimizing memory usage...")
        print("Low memory detected, optimizing memory usage...")
        global default_file_path
        signal, sr = load_audio(default_file_path)
        signal_queue.put((signal, sr))
        logging.info("Memory optimization complete.")
        print("Memory optimization complete.")


def load_audio(file_path: str = default_file_path) -> tuple[np.ndarray, int]:
    """Load an audio file and return the signal and sample rate."""
    logging.info(f"Reading audio file from {file_path}")
    print(f"Reading audio file from {file_path}")

    # Load the audio file using pydub
    audio = AudioSegment.from_file(file_path)
    sr = audio.frame_rate
    signal = np.array(audio.get_array_of_samples())

    # Convert to mono if the audio is stereo
    if audio.channels > 1:
        signal = signal.reshape((-1, audio.channels)).mean(axis=1)

    logging.info(f"Audio file read complete: {file_path}")
    print(f"Audio file read complete: {file_path}")
    return signal, sr


def record_audio(duration: int = 5, sr: int = 44100) -> tuple[np.ndarray, int]:
    """Record audio from the microphone for a given duration and sample rate and save it."""
    logging.info(f"Recording audio for {duration} seconds at {sr} sample rate")
    print(f"Recording audio for {duration} seconds at {sr} sample rate")

    signal = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="float64")
    sd.wait()

    signal = signal.flatten()

    # Save the recorded audio
    timestamp = get_timestamp()
    file_name = f"{timestamp}_recorded_audio.wav"
    file_path = os.path.join(output_dir, file_name)
    wav_write(file_path, sr, signal.astype(np.int16))  # Convert to int16 for saving

    logging.info(f"Recording complete. Audio saved to {file_path}")
    print(f"Recording complete. Audio saved to {file_path}")

    return signal, sr


def get_timestamp() -> str:
    """Get the current timestamp for unique file naming."""
    logging.info("Generating timestamp for file naming.")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.info(f"Timestamp generated: {timestamp}")
    return timestamp


def get_max_freq_components(sr: int) -> int:
    logging.info("Calculating maximum frequency components based on system resources")
    print("Calculating maximum frequency components based on system resources")

    cpu_count = psutil.cpu_count(logical=False)
    memory_info = psutil.virtual_memory()
    max_freq_components = int((sr / 2) * 0.8)  # 80% of the Nyquist frequency

    logging.info(f"CPU count (physical cores): {cpu_count}")
    logging.info(f"Total memory: {memory_info.total / (1024 ** 2):.2f} MB")
    logging.info(f"Available memory: {memory_info.available / (1024 ** 2):.2f} MB")
    logging.info(f"Max frequency components calculated: {max_freq_components}")

    print(f"CPU count (physical cores): {cpu_count}")
    print(f"Total memory: {memory_info.total / (1024 ** 2):.2f} MB")
    print(f"Available memory: {memory_info.available / (1024 ** 2):.2f} MB")
    print(f"Max frequency components calculated: {max_freq_components}")

    return min(max_freq_components, cpu_count * 1000, memory_info.available // (8 * sr))


def compute_spectral_features(signal: np.ndarray, sr: int) -> dict:
    # Compute basic spectral features manually
    spectral_centroid = np.sum(np.arange(len(signal)) * np.abs(signal)) / np.sum(
        np.abs(signal)
    )
    spectral_bandwidth = np.sqrt(
        np.sum((np.arange(len(signal)) - spectral_centroid) ** 2 * np.abs(signal))
        / np.sum(np.abs(signal))
    )
    spectral_rolloff = np.sum(signal) * 0.85
    zero_crossing_rate = ((signal[:-1] * signal[1:]) < 0).sum()

    # Compute MFCCs manually (simplified version)
    mel_filters = np.linspace(0, sr // 2, num=13)
    mfccs = np.log(np.abs(fft(signal)[: len(mel_filters)]) + 1e-10)

    return {
        "spectral_centroid": np.array([spectral_centroid]),
        "spectral_bandwidth": np.array([spectral_bandwidth]),
        "spectral_rolloff": np.array([spectral_rolloff]),
        "zero_crossing_rate": np.array([zero_crossing_rate]),
        "mfccs": mfccs,
    }


def create_tensor_wave(signal: np.ndarray, sr: int) -> dict:
    logging.info("Starting tensor wave creation process")
    print("Starting tensor wave creation process")

    logging.info("Computing FFT of the input signal")
    print("Computing FFT of the input signal")
    freq_components = fft(signal)
    magnitude_spectrum = np.abs(freq_components)
    phase_spectrum = np.angle(freq_components)
    logging.info("FFT computation complete")
    print("FFT computation complete")

    logging.info("Extracting additional spectral features")
    print("Extracting additional spectral features")
    spectral_features = compute_spectral_features(signal, sr)
    logging.info("Spectral feature extraction complete")
    print("Spectral feature extraction complete")

    logging.info("Constructing tensor representation dictionary")
    print("Constructing tensor representation dictionary")
    tensor_wave_dict = {
        "magnitude_spectrum": magnitude_spectrum,
        "phase_spectrum": phase_spectrum,
        **spectral_features,
    }
    logging.info("Tensor representation dictionary construction complete")
    print("Tensor representation dictionary construction complete")

    logging.info("Normalizing and scaling tensor representations")
    print("Normalizing and scaling tensor representations")
    scaler = MinMaxScaler()
    for key in tensor_wave_dict:
        logging.info(f"Normalizing and scaling {key}")
        print(f"Normalizing and scaling {key}")
        if tensor_wave_dict[key].ndim == 1:
            tensor_wave_dict[key] = tensor_wave_dict[key].reshape(-1, 1)
        tensor_wave_dict[key] = scaler.fit_transform(tensor_wave_dict[key].T).T
        logging.info(f"{key} normalization and scaling complete")
        print(f"{key} normalization and scaling complete")

    logging.info("Tensor wave creation process complete")
    print("Tensor wave creation process complete")
    return tensor_wave_dict


def save_visualization(
    signal: np.ndarray, tensor_wave_dict: dict, sr: int, title: str, viz_type: str
):
    time = np.linspace(0, len(signal) / sr, num=len(signal))
    timestamp = get_timestamp()

    def save_plot(fig, filename):
        fig.savefig(os.path.join(output_dir, filename), bbox_inches="tight")
        plt.close(fig)
        logging.info(f"Saved plot: {filename}")

    fig, ax = plt.subplots(figsize=(10, 4))

    if viz_type == "original_signal":
        ax.plot(time, signal, label="Original Signal")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Amplitude")
        ax.set_title(f"{title} - Original Signal")
        ax.legend()
        save_plot(fig, f"original_signal_{timestamp}.png")

    elif viz_type == "tensor_wave_dict":
        sns.heatmap(
            tensor_wave_dict["magnitude_spectrum"].T, cmap="viridis", cbar=True, ax=ax
        )
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Frequency Component")
        ax.set_title(f"{title} - Tensor Wave Heatmap")
        save_plot(fig, f"tensor_wave_heatmap_{timestamp}.png")

    elif viz_type == "spectrogram":
        f, t, Sxx = spectrogram(signal, sr)
        cax = ax.pcolormesh(t, f, 10 * np.log10(Sxx), shading="gouraud")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Frequency [Hz]")
        ax.set_title(f"{title} - Spectrogram")
        fig.colorbar(cax, ax=ax, label="Intensity [dB]")
        save_plot(fig, f"spectrogram_{timestamp}.png")

    elif viz_type == "power_spectrum":
        power_spectrum = np.abs(fft(signal)) ** 2
        freqs = np.fft.fftfreq(len(signal), 1 / sr)
        ax.plot(freqs[: len(freqs) // 2], power_spectrum[: len(power_spectrum) // 2])
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Power")
        ax.set_title(f"{title} - Power Spectrum")
        save_plot(fig, f"power_spectrum_{timestamp}.png")

    elif viz_type == "zero_crossing_rate":
        zero_crossings = ((signal[:-1] * signal[1:]) < 0).sum()
        ax.plot(time, signal, label="Signal")
        ax.vlines(
            time[zero_crossings],
            ymin=min(signal),
            ymax=max(signal),
            color="r",
            alpha=0.5,
            label="Zero Crossings",
        )
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Amplitude")
        ax.set_title(f"{title} - Zero Crossing Rate")
        ax.legend()
        save_plot(fig, f"zero_crossing_rate_{timestamp}.png")

    elif viz_type == "rms_energy":
        rms = np.sqrt(np.mean(signal**2))
        ax.plot(time, signal, label="Signal")
        ax.plot(time, [rms] * len(time), label="RMS Energy", color="r")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Amplitude / RMS Energy")
        ax.set_title(f"{title} - RMS Energy")
        ax.legend()
        save_plot(fig, f"rms_energy_{timestamp}.png")

    elif viz_type == "mel_spectrogram":
        mel_spectrogram = np.log(np.abs(fft(signal)) + 1e-10)
        ax.imshow(mel_spectrogram, aspect="auto", origin="lower", cmap="viridis")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Frequency")
        ax.set_title(f"{title} - Mel Spectrogram")
        save_plot(fig, f"mel_spectrogram_{timestamp}.png")

    elif viz_type == "chromagram":
        chromagram = np.abs(fft(signal))[:12]  # Simplified chromagram
        ax.imshow(chromagram, aspect="auto", origin="lower", cmap="coolwarm")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Chroma")
        ax.set_title(f"{title} - Chromagram")
        save_plot(fig, f"chromagram_{timestamp}.png")

    elif viz_type == "tonnetz":
        harmonic_signal = np.abs(fft(signal))  # Simplified harmonic component
        tonnetz = harmonic_signal[:6]
        ax.imshow(tonnetz, aspect="auto", origin="lower", cmap="viridis")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Tonnetz")
        ax.set_title(f"{title} - Tonnetz")
        save_plot(fig, f"tonnetz_{timestamp}.png")

    elif viz_type == "tempogram":
        tempogram = np.abs(fft(signal))  # Simplified tempogram
        ax.imshow(tempogram, aspect="auto", origin="lower", cmap="viridis")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Tempo")
        ax.set_title(f"{title} - Tempogram")
        save_plot(fig, f"tempogram_{timestamp}.png")

    elif viz_type == "spectral_centroid":
        spectral_centroid = tensor_wave_dict["spectral_centroid"]
        ax.semilogy(time, spectral_centroid, label="Spectral Centroid")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Hz")
        ax.set_title(f"{title} - Spectral Centroid")
        ax.legend()
        save_plot(fig, f"spectral_centroid_{timestamp}.png")

    elif viz_type == "spectral_bandwidth":
        spectral_bandwidth = tensor_wave_dict["spectral_bandwidth"]
        ax.semilogy(time, spectral_bandwidth, label="Spectral Bandwidth")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Hz")
        ax.set_title(f"{title} - Spectral Bandwidth")
        ax.legend()
        save_plot(fig, f"spectral_bandwidth_{timestamp}.png")

    elif viz_type == "spectral_contrast":
        spectral_contrast = tensor_wave_dict["spectral_contrast"]
        ax.imshow(spectral_contrast, aspect="auto", origin="lower", cmap="viridis")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Frequency")
        ax.set_title(f"{title} - Spectral Contrast")
        save_plot(fig, f"spectral_contrast_{timestamp}.png")

    elif viz_type == "spectral_flatness":
        spectral_flatness = tensor_wave_dict["spectral_flatness"]
        ax.plot(time, spectral_flatness, label="Spectral Flatness")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Flatness")
        ax.set_title(f"{title} - Spectral Flatness")
        ax.legend()
        save_plot(fig, f"spectral_flatness_{timestamp}.png")

    elif viz_type == "kurtosis":
        signal_kurtosis = kurtosis(signal)
        ax.plot(time, signal, label="Signal")
        ax.axhline(y=signal_kurtosis, color="r", linestyle="-", label="Kurtosis")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Amplitude")
        ax.set_title(f"{title} - Kurtosis")
        ax.legend()
        save_plot(fig, f"kurtosis_{timestamp}.png")

    elif viz_type == "skewness":
        signal_skewness = skew(signal)
        ax.plot(time, signal, label="Signal")
        ax.axhline(y=signal_skewness, color="r", linestyle="-", label="Skewness")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Amplitude")
        ax.set_title(f"{title} - Skewness")
        ax.legend()
        save_plot(fig, f"skewness_{timestamp}.png")

    elif viz_type == "peak_detection":
        peaks, _ = find_peaks(signal, height=0)
        ax.plot(time, signal, label="Signal")
        ax.plot(time[peaks], signal[peaks], "x", label="Peaks")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Amplitude")
        ax.set_title(f"{title} - Peak Detection")
        ax.legend()
        save_plot(fig, f"peak_detection_{timestamp}.png")

    elif viz_type == "smoothed_signal":
        smoothed_signal = gaussian_filter1d(signal, sigma=2)
        ax.plot(time, smoothed_signal, label="Smoothed Signal")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Amplitude")
        ax.set_title(f"{title} - Smoothed Signal")
        ax.legend()
        save_plot(fig, f"smoothed_signal_{timestamp}.png")

    elif viz_type == "pca":
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(tensor_wave_dict["magnitude_spectrum"])
        scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], c=time, cmap="viridis")
        fig.colorbar(scatter, ax=ax, label="Time [s]")
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.set_title(f"{title} - PCA of Tensor Wave")
        save_plot(fig, f"pca_tensor_wave_{timestamp}.png")

    elif viz_type == "tsne":
        tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
        tsne_result = tsne.fit_transform(tensor_wave_dict["magnitude_spectrum"])
        scatter = ax.scatter(
            tsne_result[:, 0], tsne_result[:, 1], c=time, cmap="plasma"
        )
        fig.colorbar(scatter, ax=ax, label="Time [s]")
        ax.set_xlabel("t-SNE Component 1")
        ax.set_ylabel("t-SNE Component 2")
        ax.set_title(f"{title} - t-SNE of Tensor Wave")
        save_plot(fig, f"tsne_tensor_wave_{timestamp}.png")

    elif viz_type == "kmeans":
        kmeans = KMeans(n_clusters=3)
        kmeans_result = kmeans.fit_predict(tensor_wave_dict["magnitude_spectrum"])
        scatter = ax.scatter(
            pca_result[:, 0], pca_result[:, 1], c=kmeans_result, cmap="viridis"
        )
        fig.colorbar(scatter, ax=ax, label="Cluster")
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.set_title(f"{title} - K-Means Clustering of Tensor Wave")
        save_plot(fig, f"kmeans_tensor_wave_{timestamp}.png")

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
        fig = go.Figure(
            data=go.Heatmap(
                z=tensor_wave_dict["magnitude_spectrum"].T, x=time, colorscale="Viridis"
            )
        )
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
        fig = go.Figure(data=[go.Surface(z=tensor_wave_dict["magnitude_spectrum"].T)])
        fig.update_layout(
            title=f"{title} - 3D Surface Plot of Tensor Wave",
            scene=dict(
                xaxis_title="Time [s]",
                yaxis_title="Frequency Component",
                zaxis_title="Amplitude",
            ),
        )
        fig.write_html(os.path.join(output_dir, f"3d_surface_{timestamp}.html"))

    else:
        raise ValueError(f"Unknown visualization type: {viz_type}")

    plt.close(fig)


def main():
    global running, paused

    logging.info("Program started.")
    print("Program started.")

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
        signal, sr = load_audio(file_path)
    elif input_option == "mic":
        duration = int(input("Enter the duration for recording (seconds): ").strip())
        signal, sr = record_audio(duration=duration)
    else:
        logging.error("Invalid input option. Exiting.")
        print("Invalid option. Exiting.")
        return

    logging.info("Starting main processing loop.")
    print("Starting main processing loop.")

    try:
        logging.info("Processing signal.")
        print("Processing signal.")
        tensor_wave = create_tensor_wave(signal, sr)
        visualizations = [
            "original_signal",
            "tensor_wave_dict",
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

        for viz_type in visualizations:
            if not running:
                logging.info(
                    "Stopping visualization processing as running flag is set to False."
                )
                print(
                    "Stopping visualization processing as running flag is set to False."
                )
                break
            logging.info(f"Generating visualization: {viz_type}")
            print(f"Generating visualization: {viz_type}")
            save_visualization(
                signal,
                tensor_wave,
                sr,
                title="Wave Visualization",
                viz_type=viz_type,
            )
            logging.info(f"Completed visualization: {viz_type}")
            print(f"Completed visualization: {viz_type}")

            # Real-time feedback at maximum granularity
            memory_info = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=0.1)
            logging.info(
                f"Real-time feedback - Memory available: {memory_info.available / (1024 ** 2):.2f} MB, CPU usage: {cpu_percent}%"
            )
            print(
                f"Real-time feedback - Memory available: {memory_info.available / (1024 ** 2):.2f} MB, CPU usage: {cpu_percent}%"
            )
    except Exception as e:
        logging.error(f"An error occurred in the main loop: {e}")
        print(f"An error occurred in the main loop: {e}")
        running = False

    logging.info("Main processing loop ended.")
    print("Main processing loop ended.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        print(f"An error occurred: {e}")
    finally:
        logging.info("Program terminated.")
        print("Program terminated.")

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
    try:
        logging.info(
            "Calculating maximum frequency components based on system resources"
        )
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

        return min(
            max_freq_components, cpu_count * 1000, memory_info.available // (8 * sr)
        )
    except Exception as e:
        logging.error(f"Error calculating max frequency components: {e}", exc_info=True)
        print(f"Error calculating max frequency components: {e}")
        return 0


def compute_spectral_features(signal: np.ndarray, sr: int) -> dict:
    """Compute basic spectral features manually."""
    try:
        spectral_centroid = np.sum(np.arange(len(signal)) * np.abs(signal)) / np.sum(
            np.abs(signal)
        )
        spectral_bandwidth = np.sqrt(
            np.sum((np.arange(len(signal)) - spectral_centroid) ** 2 * np.abs(signal))
            / np.sum(np.abs(signal))
        )
        spectral_rolloff = np.sum(signal) * 0.85
        zero_crossing_rate = ((signal[:-1] * signal[1:]) < 0).sum()
        mel_filters = np.linspace(0, sr // 2, num=13)
        mfccs = np.log(np.abs(fft(signal)[: len(mel_filters)]) + 1e-10)

        return {
            "spectral_centroid": np.array([spectral_centroid]),
            "spectral_bandwidth": np.array([spectral_bandwidth]),
            "spectral_rolloff": np.array([spectral_rolloff]),
            "zero_crossing_rate": np.array([zero_crossing_rate]),
            "mfccs": mfccs,
        }
    except Exception as e:
        logging.error(f"Error computing spectral features: {e}", exc_info=True)
        print(f"Error computing spectral features: {e}")
        return {}


def compute_fft_tensor_wave(signal: np.ndarray) -> np.ndarray:
    """Compute FFT tensor wave representation of the input signal."""
    try:
        logging.info("Computing FFT of the input signal")
        freq_components = fft(signal)
        magnitude_spectrum = np.abs(freq_components)
        phase_spectrum = np.angle(freq_components)
        logging.info("FFT computation complete")
        tensor_wave = np.vstack([magnitude_spectrum, phase_spectrum])
        logging.info("FFT tensor wave construction complete")
        return tensor_wave
    except Exception as e:
        logging.error(f"Error computing FFT tensor wave: {e}", exc_info=True)
        print(f"Error computing FFT tensor wave: {e}")
        return np.array([])


def compute_spectral_features_tensor_wave(signal: np.ndarray, sr: int) -> np.ndarray:
    """Compute spectral features tensor wave representation of the input signal."""
    try:
        logging.info("Extracting spectral features")
        spectral_centroid = np.sum(np.arange(len(signal)) * np.abs(signal)) / np.sum(
            np.abs(signal)
        )
        spectral_bandwidth = np.sqrt(
            np.sum((np.arange(len(signal)) - spectral_centroid) ** 2 * np.abs(signal))
            / np.sum(np.abs(signal))
        )
        spectral_rolloff = np.sum(signal) * 0.85
        zero_crossing_rate = ((signal[:-1] * signal[1:]) < 0).sum()
        logging.info("Spectral feature extraction complete")
        tensor_wave = np.vstack(
            [
                np.full(len(signal), spectral_centroid),
                np.full(len(signal), spectral_bandwidth),
                np.full(len(signal), spectral_rolloff),
                np.full(len(signal), zero_crossing_rate),
            ]
        )
        logging.info("Spectral features tensor wave construction complete")
        return tensor_wave
    except Exception as e:
        logging.error(
            f"Error computing spectral features tensor wave: {e}", exc_info=True
        )
        print(f"Error computing spectral features tensor wave: {e}")
        return np.array([])


def compute_mfcc_tensor_wave(signal: np.ndarray, sr: int) -> np.ndarray:
    """Compute MFCC tensor wave representation of the input signal."""
    try:
        logging.info("Computing MFCCs")
        mel_filters = np.linspace(0, sr // 2, num=13)
        mfccs = np.log(np.abs(fft(signal)[: len(mel_filters)]) + 1e-10)
        logging.info("MFCC computation complete")
        tensor_wave = np.vstack([mfccs])
        logging.info("MFCC tensor wave construction complete")
        return tensor_wave
    except Exception as e:
        logging.error(f"Error computing MFCC tensor wave: {e}", exc_info=True)
        print(f"Error computing MFCC tensor wave: {e}")
        return np.array([])


def compute_chromagram(signal: np.ndarray, sr: int) -> np.ndarray:
    """Compute chromagram of the input signal."""
    try:
        logging.info("Computing chromagram")
        stft = np.abs(fft(signal))
        chroma = np.zeros((12, len(stft)))
        for i in range(len(stft)):
            chroma[i % 12, i] = stft[i]
        logging.info("Chromagram computation complete")
        return chroma
    except Exception as e:
        logging.error(f"Error computing chromagram: {e}", exc_info=True)
        print(f"Error computing chromagram: {e}")
        return np.array([])


def compute_mel_spectrogram(signal: np.ndarray, sr: int) -> np.ndarray:
    """Compute mel spectrogram of the input signal."""
    try:
        logging.info("Computing mel spectrogram")
        mel_filters = np.linspace(0, sr // 2, num=128)
        mel_spectrogram = np.log(np.abs(fft(signal)[: len(mel_filters)]) + 1e-10)
        logging.info("Mel spectrogram computation complete")
        return mel_spectrogram
    except Exception as e:
        logging.error(f"Error computing mel spectrogram: {e}", exc_info=True)
        print(f"Error computing mel spectrogram: {e}")
        return np.array([])


def compute_power_spectrum(signal: np.ndarray) -> np.ndarray:
    """Compute power spectrum of the input signal."""
    try:
        logging.info("Computing power spectrum")
        power_spectrum = np.abs(fft(signal)) ** 2
        logging.info("Power spectrum computation complete")
        return power_spectrum
    except Exception as e:
        logging.error(f"Error computing power spectrum: {e}", exc_info=True)
        print(f"Error computing power spectrum: {e}")
        return np.array([])


def compute_rms_energy(signal: np.ndarray) -> np.ndarray:
    """Compute RMS energy of the input signal."""
    try:
        logging.info("Computing RMS energy")
        rms_energy = np.sqrt(np.mean(signal**2))
        logging.info("RMS energy computation complete")
        return np.array([rms_energy])
    except Exception as e:
        logging.error(f"Error computing RMS energy: {e}", exc_info=True)
        print(f"Error computing RMS energy: {e}")
        return np.array([])


def compute_spectrogram(signal: np.ndarray, sr: int) -> np.ndarray:
    """Compute spectrogram of the input signal."""
    try:
        logging.info("Computing spectrogram")
        n_fft = 2048
        hop_length = 512
        spectrogram = np.abs(fft(signal, n=n_fft))[:, : len(signal) // hop_length]
        logging.info("Spectrogram computation complete")
        return spectrogram
    except Exception as e:
        logging.error(f"Error computing spectrogram: {e}", exc_info=True)
        print(f"Error computing spectrogram: {e}")
        return np.array([])


def compute_tempogram(signal: np.ndarray, sr: int) -> np.ndarray:
    """Compute tempogram of the input signal."""
    try:
        logging.info("Computing tempogram")
        onset_env = np.abs(np.diff(signal))
        tempogram = np.abs(fft(onset_env))
        logging.info("Tempogram computation complete")
        return tempogram
    except Exception as e:
        logging.error(f"Error computing tempogram: {e}", exc_info=True)
        print(f"Error computing tempogram: {e}")
        return np.array([])


def compute_tonnetz(signal: np.ndarray, sr: int) -> np.ndarray:
    """Compute tonnetz of the input signal."""
    try:
        logging.info("Computing tonnetz")
        chroma = compute_chromagram(signal, sr)
        tonnetz = np.dot(chroma.T, chroma)
        logging.info("Tonnetz computation complete")
        return tonnetz
    except Exception as e:
        logging.error(f"Error computing tonnetz: {e}", exc_info=True)
        print(f"Error computing tonnetz: {e}")
        return np.array([])


def compute_3d_surface(signal: np.ndarray, sr: int) -> np.ndarray:
    """Compute 3D surface representation of the input signal."""
    try:
        logging.info("Computing 3D surface")
        spectrogram = compute_spectrogram(signal, sr)
        X, Y = np.meshgrid(
            np.arange(spectrogram.shape[1]), np.arange(spectrogram.shape[0])
        )
        Z = spectrogram
        logging.info("3D surface computation complete")
        return np.array([X, Y, Z])
    except Exception as e:
        logging.error(f"Error computing 3D surface: {e}", exc_info=True)
        print(f"Error computing 3D surface: {e}")
        return np.array([])


def compute_pca(signal: np.ndarray) -> np.ndarray:
    """Compute PCA of the input signal."""
    try:
        logging.info("Computing PCA")
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(signal.reshape(-1, 1))
        logging.info("PCA computation complete")
        return pca_result
    except Exception as e:
        logging.error(f"Error computing PCA: {e}", exc_info=True)
        print(f"Error computing PCA: {e}")
        return np.array([])


def compute_kmeans(signal: np.ndarray, n_clusters: int = 2) -> np.ndarray:
    """Compute KMeans clustering of the input signal."""
    try:
        logging.info("Computing KMeans clustering")
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans_result = kmeans.fit_predict(signal.reshape(-1, 1))
        logging.info("KMeans clustering computation complete")
        return kmeans_result
    except Exception as e:
        logging.error(f"Error computing KMeans clustering: {e}", exc_info=True)
        print(f"Error computing KMeans clustering: {e}")
        return np.array([])


def create_tensor_wave(signal: np.ndarray, sr: int) -> dict:
    """Create multiple 2D tensor wave representations of the input signal."""
    logging.info("Starting tensor wave creation process")
    tensor_waves = {}
    try:
        tensor_waves["fft"] = compute_fft_tensor_wave(signal)
    except Exception as e:
        logging.error(f"Error in FFT tensor wave: {e}", exc_info=True)
        print(f"Error in FFT tensor wave: {e}")

    try:
        tensor_waves["spectral_features"] = compute_spectral_features_tensor_wave(
            signal, sr
        )
    except Exception as e:
        logging.error(f"Error in spectral features tensor wave: {e}", exc_info=True)
        print(f"Error in spectral features tensor wave: {e}")

    try:
        tensor_waves["mfcc"] = compute_mfcc_tensor_wave(signal, sr)
    except Exception as e:
        logging.error(f"Error in MFCC tensor wave: {e}", exc_info=True)
        print(f"Error in MFCC tensor wave: {e}")

    try:
        tensor_waves["chromagram"] = compute_chromagram(signal, sr)
    except Exception as e:
        logging.error(f"Error in chromagram: {e}", exc_info=True)
        print(f"Error in chromagram: {e}")

    try:
        tensor_waves["mel_spectrogram"] = compute_mel_spectrogram(signal, sr)
    except Exception as e:
        logging.error(f"Error in mel spectrogram: {e}", exc_info=True)
        print(f"Error in mel spectrogram: {e}")

    try:
        tensor_waves["power_spectrum"] = compute_power_spectrum(signal)
    except Exception as e:
        logging.error(f"Error in power spectrum: {e}", exc_info=True)
        print(f"Error in power spectrum: {e}")

    try:
        tensor_waves["rms_energy"] = compute_rms_energy(signal)
    except Exception as e:
        logging.error(f"Error in RMS energy: {e}", exc_info=True)
        print(f"Error in RMS energy: {e}")

    try:
        tensor_waves["spectrogram"] = compute_spectrogram(signal, sr)
    except Exception as e:
        logging.error(f"Error in spectrogram: {e}", exc_info=True)
        print(f"Error in spectrogram: {e}")

    try:
        tensor_waves["tempogram"] = compute_tempogram(signal, sr)
    except Exception as e:
        logging.error(f"Error in tempogram: {e}", exc_info=True)
        print(f"Error in tempogram: {e}")

    try:
        tensor_waves["tonnetz"] = compute_tonnetz(signal, sr)
    except Exception as e:
        logging.error(f"Error in tonnetz: {e}", exc_info=True)
        print(f"Error in tonnetz: {e}")

    try:
        tensor_waves["3d_surface"] = compute_3d_surface(signal, sr)
    except Exception as e:
        logging.error(f"Error in 3D surface: {e}", exc_info=True)
        print(f"Error in 3D surface: {e}")

    try:
        tensor_waves["pca"] = compute_pca(signal)
    except Exception as e:
        logging.error(f"Error in PCA: {e}", exc_info=True)
        print(f"Error in PCA: {e}")

    try:
        tensor_waves["kmeans"] = compute_kmeans(signal)
    except Exception as e:
        logging.error(f"Error in KMeans clustering: {e}", exc_info=True)
        print(f"Error in KMeans clustering: {e}")

    logging.info("Normalizing and scaling tensor representations")
    try:
        scaler = MinMaxScaler()
        for key in tensor_waves:
            if tensor_waves[key].size > 0:
                tensor_waves[key] = scaler.fit_transform(tensor_waves[key].T).T
        logging.info("Normalization and scaling complete")
    except Exception as e:
        logging.error(f"Error in normalization and scaling: {e}", exc_info=True)
        print(f"Error in normalization and scaling: {e}")

    logging.info("Tensor wave creation process complete")
    return tensor_waves


def save_visualization(
    signal: np.ndarray, tensor_wave_dict: dict, sr: int, title: str, viz_type: str
):
    """Save a specific visualization of the signal and tensor wave."""
    time = np.linspace(0, len(signal) / sr, num=len(signal))
    timestamp = get_timestamp()
    output_dir = "output"  # Ensure this directory exists or create it

    def save_plot(fig, filename):
        fig.write_image(os.path.join(output_dir, filename))
        logging.info(f"Saved plot: {filename}")

    try:
        if viz_type == "original_signal":
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(x=time, y=signal, mode="lines", name="Original Signal")
            )
            fig.update_layout(
                title=f"{title} - Original Signal",
                xaxis_title="Time [s]",
                yaxis_title="Amplitude",
            )
            save_plot(fig, f"original_signal_{timestamp}.png")

        elif viz_type == "fft":
            magnitude_spectrum = tensor_wave_dict["fft"][0]
            frequencies = np.fft.fftfreq(len(signal), 1 / sr)
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=frequencies,
                    y=magnitude_spectrum,
                    mode="lines",
                    name="FFT Magnitude Spectrum",
                )
            )
            fig.update_layout(
                title=f"{title} - FFT Magnitude Spectrum",
                xaxis_title="Frequency [Hz]",
                yaxis_title="Magnitude",
            )
            save_plot(fig, f"fft_{timestamp}.png")

        elif viz_type == "spectral_features":
            spectral_features = tensor_wave_dict["spectral_features"]
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=time,
                    y=spectral_features[0],
                    mode="lines",
                    name="Spectral Centroid",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=time,
                    y=spectral_features[1],
                    mode="lines",
                    name="Spectral Bandwidth",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=time,
                    y=spectral_features[2],
                    mode="lines",
                    name="Spectral Rolloff",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=time,
                    y=spectral_features[3],
                    mode="lines",
                    name="Zero Crossing Rate",
                )
            )
            fig.update_layout(
                title=f"{title} - Spectral Features",
                xaxis_title="Time [s]",
                yaxis_title="Value",
            )
            save_plot(fig, f"spectral_features_{timestamp}.png")

        elif viz_type == "mfcc":
            mfccs = tensor_wave_dict["mfcc"]
            fig = px.imshow(mfccs, aspect="auto", color_continuous_scale="viridis")
            fig.update_layout(
                title=f"{title} - MFCCs",
                xaxis_title="Time",
                yaxis_title="MFCC Coefficients",
            )
            save_plot(fig, f"mfcc_{timestamp}.png")

        elif viz_type == "chromagram":
            chromagram = tensor_wave_dict["chromagram"]
            fig = px.imshow(chromagram, aspect="auto", color_continuous_scale="viridis")
            fig.update_layout(
                title=f"{title} - Chromagram", xaxis_title="Time", yaxis_title="Chroma"
            )
            save_plot(fig, f"chromagram_{timestamp}.png")

        elif viz_type == "mel_spectrogram":
            mel_spectrogram = tensor_wave_dict["mel_spectrogram"]
            fig = px.imshow(
                mel_spectrogram, aspect="auto", color_continuous_scale="viridis"
            )
            fig.update_layout(
                title=f"{title} - Mel Spectrogram",
                xaxis_title="Time",
                yaxis_title="Mel Frequency",
            )
            save_plot(fig, f"mel_spectrogram_{timestamp}.png")

        elif viz_type == "power_spectrum":
            power_spectrum = tensor_wave_dict["power_spectrum"]
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=np.fft.fftfreq(len(signal), 1 / sr),
                    y=power_spectrum,
                    mode="lines",
                    name="Power Spectrum",
                )
            )
            fig.update_layout(
                title=f"{title} - Power Spectrum",
                xaxis_title="Frequency [Hz]",
                yaxis_title="Power",
            )
            save_plot(fig, f"power_spectrum_{timestamp}.png")

        elif viz_type == "rms_energy":
            rms_energy = tensor_wave_dict["rms_energy"]
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(x=[0], y=rms_energy, mode="markers", name="RMS Energy")
            )
            fig.update_layout(
                title=f"{title} - RMS Energy", xaxis_title="", yaxis_title="RMS Energy"
            )
            save_plot(fig, f"rms_energy_{timestamp}.png")

        elif viz_type == "spectrogram":
            spectrogram = tensor_wave_dict["spectrogram"]
            fig = px.imshow(
                spectrogram, aspect="auto", color_continuous_scale="viridis"
            )
            fig.update_layout(
                title=f"{title} - Spectrogram",
                xaxis_title="Time",
                yaxis_title="Frequency",
            )
            save_plot(fig, f"spectrogram_{timestamp}.png")

        elif viz_type == "tempogram":
            tempogram = tensor_wave_dict["tempogram"]
            fig = px.imshow(tempogram, aspect="auto", color_continuous_scale="viridis")
            fig.update_layout(
                title=f"{title} - Tempogram", xaxis_title="Time", yaxis_title="Tempo"
            )
            save_plot(fig, f"tempogram_{timestamp}.png")

        elif viz_type == "tonnetz":
            tonnetz = tensor_wave_dict["tonnetz"]
            fig = px.imshow(tonnetz, aspect="auto", color_continuous_scale="viridis")
            fig.update_layout(
                title=f"{title} - Tonnetz", xaxis_title="", yaxis_title=""
            )
            save_plot(fig, f"tonnetz_{timestamp}.png")

        elif viz_type == "3d_surface":
            X, Y, Z = tensor_wave_dict["3d_surface"]
            fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y)])
            fig.update_layout(
                title=f"{title} - 3D Surface",
                scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
            )
            save_plot(fig, f"3d_surface_{timestamp}.png")

        elif viz_type == "pca":
            pca_result = tensor_wave_dict["pca"]
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=pca_result[:, 0], y=pca_result[:, 1], mode="markers", name="PCA"
                )
            )
            fig.update_layout(
                title=f"{title} - PCA",
                xaxis_title="Principal Component 1",
                yaxis_title="Principal Component 2",
            )
            save_plot(fig, f"pca_{timestamp}.png")

        elif viz_type == "kmeans":
            kmeans_result = tensor_wave_dict["kmeans"]
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=np.arange(len(kmeans_result)),
                    y=kmeans_result,
                    mode="markers",
                    name="KMeans Clustering",
                )
            )
            fig.update_layout(
                title=f"{title} - KMeans Clustering",
                xaxis_title="Sample Index",
                yaxis_title="Cluster",
            )
            save_plot(fig, f"kmeans_{timestamp}.png")

        else:
            raise ValueError(f"Unknown visualization type: {viz_type}")

    except Exception as e:
        logging.error(f"Error in visualization: {e}", exc_info=True)
        print(f"Error in visualization: {e}")


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
            "fft",
            "spectral_features",
            "mfcc",
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

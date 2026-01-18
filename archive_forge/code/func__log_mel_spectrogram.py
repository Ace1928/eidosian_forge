import math
import torch
def _log_mel_spectrogram(data, audio_sample_rate=8000, log_offset=0.0, window_length_secs=0.025, hop_length_secs=0.01, **kwargs):
    window_length_samples = int(round(audio_sample_rate * window_length_secs))
    hop_length_samples = int(round(audio_sample_rate * hop_length_secs))
    fft_length = 2 ** int(math.ceil(math.log(window_length_samples) / math.log(2.0)))
    spectrogram = _stft_magnitude(data, fft_length=fft_length, hop_length=hop_length_samples, window_length=window_length_samples)
    mel_spectrogram = torch.matmul(spectrogram, _spectrogram_to_mel_matrix(num_spectrogram_bins=spectrogram.shape[1], audio_sample_rate=audio_sample_rate, **kwargs).to(spectrogram))
    return torch.log(mel_spectrogram + log_offset)
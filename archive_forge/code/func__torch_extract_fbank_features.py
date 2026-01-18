from typing import List, Optional, Union
import numpy as np
from ... import is_torch_available
from ...audio_utils import mel_filter_bank, spectrogram, window_function
from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import TensorType, logging
def _torch_extract_fbank_features(self, waveform: np.array) -> np.ndarray:
    """
        Compute the log-mel spectrogram of the provided audio using the PyTorch STFT implementation.
        """
    waveform = torch.from_numpy(waveform).type(torch.float32)
    window = torch.hann_window(self.n_fft)
    stft = torch.stft(waveform, self.n_fft, self.hop_length, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2
    mel_filters = torch.from_numpy(self.mel_filters).type(torch.float32)
    mel_spec = mel_filters.T @ magnitudes
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec.numpy()
from typing import List, Optional, Union
import numpy as np
from ... import is_torch_available
from ...audio_utils import mel_filter_bank, spectrogram, window_function
from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import TensorType, logging
def _np_extract_fbank_features(self, waveform: np.array) -> np.ndarray:
    """
        Compute the log-mel spectrogram of the provided audio, gives similar results to Whisper's original torch
        implementation with 1e-5 tolerance.
        """
    log_spec = spectrogram(waveform, window_function(self.n_fft, 'hann'), frame_length=self.n_fft, hop_length=self.hop_length, power=2.0, mel_filters=self.mel_filters, log_mel='log10')
    log_spec = log_spec[:, :-1]
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec
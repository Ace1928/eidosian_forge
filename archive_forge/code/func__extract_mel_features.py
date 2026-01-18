import warnings
from typing import Any, Dict, List, Optional, Union
import numpy as np
from ...audio_utils import mel_filter_bank, optimal_fft_length, spectrogram, window_function
from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import PaddingStrategy, TensorType, logging
def _extract_mel_features(self, one_waveform: np.ndarray) -> np.ndarray:
    """
        Extracts log-mel filterbank features for one waveform array (unbatched).
        """
    log_mel_spec = spectrogram(one_waveform, window=self.window, frame_length=self.sample_size, hop_length=self.sample_stride, fft_length=self.n_fft, mel_filters=self.mel_filters, mel_floor=self.mel_floor, log_mel='log10')
    return log_mel_spec.T
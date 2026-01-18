from typing import List, Optional, Union
import numpy as np
from ...utils import is_torch_available
from ...audio_utils import mel_filter_bank, spectrogram, window_function
from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...feature_extraction_utils import BatchFeature
from ...utils import PaddingStrategy, TensorType, logging
def _extract_fbank_features(self, waveform: np.ndarray) -> np.ndarray:
    """
        Get mel-filter bank features using TorchAudio. Note that TorchAudio requires 16-bit signed integers as inputs
        and hence the waveform should not be normalized before feature extraction.
        """
    if len(waveform.shape) == 2:
        waveform = waveform[0]
    waveform = np.squeeze(waveform) * 2 ** 15
    features = spectrogram(waveform, self.window, frame_length=400, hop_length=160, fft_length=512, power=2.0, center=False, preemphasis=0.97, mel_filters=self.mel_filters, log_mel='log', mel_floor=1.192092955078125e-07, remove_dc_offset=True).T
    return features
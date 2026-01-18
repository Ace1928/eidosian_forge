from typing import List, Optional, Union
import numpy as np
from ....audio_utils import mel_filter_bank, optimal_fft_length, spectrogram, window_function
from ....feature_extraction_sequence_utils import SequenceFeatureExtractor
from ....feature_extraction_utils import BatchFeature
from ....file_utils import PaddingStrategy, TensorType
from ....utils import logging
def _extract_mfsc_features(self, one_waveform: np.array) -> np.ndarray:
    """
        Extracts MFSC Features for one waveform vector (unbatched). Adapted from Flashlight's C++ MFSC code.
        """
    if self.win_function == 'hamming_window':
        window = window_function(window_length=self.sample_size, name=self.win_function, periodic=False)
    else:
        window = window_function(window_length=self.sample_size, name=self.win_function)
    fbanks = mel_filter_bank(num_frequency_bins=self.n_freqs, num_mel_filters=self.feature_size, min_frequency=0.0, max_frequency=self.sampling_rate / 2.0, sampling_rate=self.sampling_rate)
    msfc_features = spectrogram(one_waveform * self.frame_signal_scale, window=window, frame_length=self.sample_size, hop_length=self.sample_stride, fft_length=self.n_fft, center=False, preemphasis=self.preemphasis_coeff, mel_filters=fbanks, mel_floor=self.mel_floor, log_mel='log')
    return msfc_features.T
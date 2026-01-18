from typing import List, Optional, Union
import numpy as np
from ....audio_utils import mel_filter_bank, optimal_fft_length, spectrogram, window_function
from ....feature_extraction_sequence_utils import SequenceFeatureExtractor
from ....feature_extraction_utils import BatchFeature
from ....file_utils import PaddingStrategy, TensorType
from ....utils import logging
def _normalize_one(self, x, input_length, padding_value):
    if self.normalize_means:
        mean = x[:input_length].mean(axis=0)
        x = np.subtract(x, mean)
    if self.normalize_vars:
        std = x[:input_length].std(axis=0)
        x = np.divide(x, std)
    if input_length < x.shape[0]:
        x[input_length:] = padding_value
    x = x.astype(np.float32)
    return x
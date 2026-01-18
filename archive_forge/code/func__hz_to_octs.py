import math
import warnings
from typing import Optional
import torch
from torchaudio.functional.functional import _create_triangular_filterbank
def _hz_to_octs(freqs, tuning=0.0, bins_per_octave=12):
    a440 = 440.0 * 2.0 ** (tuning / bins_per_octave)
    return torch.log2(freqs / (a440 / 16))
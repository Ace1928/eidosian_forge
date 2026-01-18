import warnings
from typing import Optional, Tuple
import torch
from torchaudio._internal import module_utils as _mod_utils
from .common import AudioMetaData
def _get_bit_depth(subtype):
    if subtype not in _SUBTYPE_TO_BITS_PER_SAMPLE:
        warnings.warn(f'The {subtype} subtype is unknown to TorchAudio. As a result, the bits_per_sample attribute will be set to 0. If you are seeing this warning, please report by opening an issue on github (after checking for existing/closed ones). You may otherwise ignore this warning.')
    return _SUBTYPE_TO_BITS_PER_SAMPLE.get(subtype, 0)
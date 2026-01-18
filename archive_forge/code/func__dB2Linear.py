import math
import warnings
from typing import Optional
import torch
from torch import Tensor
from torchaudio._extension import _IS_TORCHAUDIO_EXT_AVAILABLE
def _dB2Linear(x: float) -> float:
    return math.exp(x * math.log(10) / 20.0)
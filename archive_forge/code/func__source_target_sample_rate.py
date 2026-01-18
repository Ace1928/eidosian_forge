import math
import warnings
from typing import Callable, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor
from torch.nn.modules.lazy import LazyModuleMixin
from torch.nn.parameter import UninitializedParameter
from torchaudio import functional as F
from torchaudio.functional.functional import (
def _source_target_sample_rate(orig_freq: int, speed: float) -> Tuple[int, int]:
    source_sample_rate = int(speed * orig_freq)
    target_sample_rate = int(orig_freq)
    gcd = math.gcd(source_sample_rate, target_sample_rate)
    return (source_sample_rate // gcd, target_sample_rate // gcd)
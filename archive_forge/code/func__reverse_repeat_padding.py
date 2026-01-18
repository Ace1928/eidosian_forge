from typing import Optional, List, TypeVar
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.ao.nn.intrinsic as nni
import torch.ao.nn.intrinsic.qat as nniqat
from torch._ops import ops
from torch.nn.common_types import _size_1_t
from torch.nn.modules.utils import _single, _pair, _triple
from torch.nn.utils import fuse_conv_bn_weights
from .utils import _quantize_weight, WeightedQuantizedModule
def _reverse_repeat_padding(padding: List[int]) -> List[int]:
    _reversed_padding_repeated_twice: List[int] = []
    N = len(padding)
    for idx in range(N):
        for _ in range(2):
            _reversed_padding_repeated_twice.append(padding[N - idx - 1])
    return _reversed_padding_repeated_twice
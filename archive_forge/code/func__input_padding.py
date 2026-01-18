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
def _input_padding(self, kernel_size: List[int], dilation: List[int], padding: List[int]) -> List[int]:
    res = torch.jit.annotate(List[int], [])
    for kdx in range(len(kernel_size)):
        pad = dilation[kdx] * (kernel_size[kdx] - 1) - padding[kdx]
        res.append(pad)
    return res
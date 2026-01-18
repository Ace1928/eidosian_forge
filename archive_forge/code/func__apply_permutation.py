import numbers
import warnings
import torch
import torch.nn as nn
from torch import Tensor  # noqa: F401
from torch._jit_internal import Tuple, Optional, List, Union, Dict  # noqa: F401
from torch.nn.utils.rnn import PackedSequence
from torch.ao.nn.quantized.modules.utils import _quantize_weight
def _apply_permutation(tensor: Tensor, permutation: Tensor, dim: int=1) -> Tensor:
    return tensor.index_select(dim, permutation)
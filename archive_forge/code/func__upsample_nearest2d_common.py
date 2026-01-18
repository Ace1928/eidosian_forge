import functools
import numbers
import operator
import sys
from enum import Enum
from functools import partial, reduce
from itertools import chain, product
from typing import Callable, cast, Iterable, List, Optional, Tuple, Union
import torch
import torch._prims as prims
import torch._prims_common as utils
import torch.nn.functional as F
from torch import sym_float, sym_int, Tensor
from torch._decomp import register_decomposition
from torch._higher_order_ops.out_dtype import out_dtype
from torch._prims_common import IntLike, NumberType, TensorLike, TensorSequenceType
from torch._prims_common.wrappers import (
from torch.utils import _pytree as pytree
from torch.utils._pytree import tree_map
def _upsample_nearest2d_common(input, h_indices, w_indices):
    result = aten._unsafe_index(input, (None, None, h_indices, w_indices))
    memory_format = utils.suggest_memory_format(input)
    _, n_channels, _, _ = input.shape
    if input.device.type == 'cuda' and n_channels < 4:
        memory_format = torch.contiguous_format
    result = result.contiguous(memory_format=memory_format)
    return result
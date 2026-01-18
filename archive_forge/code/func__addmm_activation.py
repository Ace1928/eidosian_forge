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
@register_decomposition(aten._addmm_activation)
@out_wrapper()
@pw_cast_for_opmath
def _addmm_activation(self: Tensor, mat1: Tensor, mat2: Tensor, beta: int=1, alpha: int=1, use_gelu: bool=False):
    out = addmm(self, mat1, mat2, beta, alpha)
    if use_gelu:
        if self.is_cuda:
            return aten.gelu(out, approximate='tanh')
        else:
            return aten.gelu(out)
    return aten.relu(out)
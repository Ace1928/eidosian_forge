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
def _index_add(x: TensorLike, dim: int, index: TensorLike, tensor: TensorLike, *, inplace: bool, alpha: NumberType=1):
    dim = utils.canonicalize_dims(x.ndim, dim)
    torch._check(index.ndim <= 1, lambda: f'Index should have dimension 1 or 0 (got {index.ndim})')
    index_size = index.size(0) if index.ndim == 1 else 1
    tensor_size = tensor.size(dim) if tensor.ndim > 0 else 1
    torch._check(tensor_size == index_size, lambda: f'Number of indices ({index_size}) should be equal to tensor.size(dim) ({tensor_size}), for dim={dim!r}')
    if alpha != 1:
        python_type = utils.dtype_to_type(x.dtype)
        torch._check(python_type == bool or utils.is_weakly_lesser_type(type(alpha), python_type), lambda: f'alpha argument of type {type(alpha)} cannot be safely cast to type {python_type}!')
        tensor = tensor * alpha
    zero_dim = x.ndim == 0
    x1 = x.unsqueeze(0) if zero_dim else x
    idx = (None,) * dim + (index,)
    index_put = aten.index_put_ if inplace else aten.index_put
    out = index_put(x1, idx, tensor, accumulate=True)
    if inplace:
        return x
    else:
        return out.squeeze(0) if zero_dim else out.contiguous()
import math
from enum import Enum
from functools import partial
from typing import List, Optional, Sequence, Tuple, Union
import torch
import torch._prims_common as utils
from torch import SymBool, SymFloat, Tensor
from torch._decomp import (
from torch._ops import OpOverload
from torch._prims import _prim_elementwise_meta, ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND
from torch._prims_common import (
from torch._prims_common.wrappers import (
from torch._refs import _broadcast_shapes, _maybe_broadcast
from torch.utils import _pytree as pytree
import torch._refs
import torch._refs.nn.functional
import torch._refs.special
@register_meta(aten._linalg_svd.default)
def _linalg_svd_meta(A: Tensor, full_matrices: bool=False, compute_uv: bool=True, driver: Optional[str]=None):
    checkIsMatrix(A, 'linalg.svd')
    checkFloatingOrComplex(A, 'linalg.svd')
    batch_dims = list(A.shape[:-2])
    m = A.shape[-2]
    n = A.shape[-1]
    k = min(m, n)
    if compute_uv:
        U_shape = batch_dims + [m, m if full_matrices else k]
        U = A.new_empty(U_shape)
        U.as_strided_(U_shape, make_contiguous_strides_for(U_shape, row_major=False))
        V_shape = batch_dims + [n if full_matrices else k, n]
        V = A.new_empty(V_shape)
        is_cuda = device_hint(A) == 'cuda'
        V.as_strided_(V_shape, make_contiguous_strides_for(V_shape, row_major=is_cuda))
    else:
        U = A.new_empty([0])
        V = A.new_empty([0])
    S = A.new_empty(batch_dims + [k], dtype=toRealValueType(A.dtype))
    return (U, S, V)
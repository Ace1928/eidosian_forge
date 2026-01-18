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
@register_meta(aten._linalg_det.default)
def _linalg_det_meta(A):
    squareCheckInputs(A, 'linalg.det')
    checkFloatingOrComplex(A, 'linalg.det')
    det = A.new_empty(A.shape[:-2])
    LU = A.new_empty(A.shape)
    LU.as_strided_(A.shape, make_contiguous_strides_for(A.shape, row_major=False))
    pivots = A.new_empty(A.shape[:-1], dtype=torch.int32)
    return (det, LU, pivots)
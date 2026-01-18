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
@register_meta(aten._linalg_solve_ex)
def _linalg_solve_ex(A: Tensor, B: Tensor, *, left: bool=True, check_errors: bool=False, result: Optional[Tensor]=None, LU: Optional[Tensor]=None, pivots: Optional[Tensor]=None, info: Optional[Tensor]=None) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    checkFloatingOrComplex(A, 'linalg.solve')
    torch._check(A.dtype == B.dtype, lambda: f'linalg.solve: Expected A and B to have the same dtype, but found A of type {A.dtype} and B of type {B.dtype} instead')
    vector_case = linalg_solve_is_vector_rhs(A, B)
    B_ = B.unsqueeze(-1) if vector_case else B
    checkInputsSolver(A, B_, left, 'linalg.solve')
    B_broad_shape, _ = _linalg_broadcast_batch_dims(B_, A)
    torch._check(left or not vector_case, lambda: 'linalg.solve: Vector broadcasting of the left hand side is not supported for left=False. In this case linalg.solve is equivalent to B / A.squeeze(-1)')
    result_shape = B_broad_shape[:-1] if vector_case else B_broad_shape
    result_ = torch.empty_strided(size=result_shape, stride=make_contiguous_strides_for(result_shape, not left), dtype=B.dtype, device=B.device)
    shape = A.shape
    ndim = A.ndim
    LU_ = torch.empty_strided(size=shape, stride=make_contiguous_strides_for(shape, False), dtype=A.dtype, device=A.device)
    pivots_ = A.new_empty(shape[:-1], dtype=torch.int32)
    info_ = A.new_empty(shape[:-2], dtype=torch.int32)
    out = (result, LU, pivots, info)
    res = (result_, LU_, pivots_, info_)
    if all((x is not None for x in out)):
        for r, o in zip(res, out):
            _maybe_resize_out(o, r.shape)
            o.as_strided_(r.shape, r.stride())
            _safe_copy_out(copy_from=r, copy_to=o, exact_dtype=False)
    return res
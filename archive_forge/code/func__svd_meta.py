import contextlib
import itertools
import operator
import weakref
from enum import Enum
from functools import partial, reduce
from typing import Any, Callable, List, Optional, Sequence, Tuple, Type, Union
import torch
import torch._prims_common as utils
import torch.library
from torch import sym_float, Tensor, TypedStorage
from torch._C import _get_default_device
from torch._prims.debug_prims import register_debug_prims
from torch._prims.rng_prims import register_rng_prims
from torch._prims_common import (
from torch._prims_common.wrappers import backwards_not_supported
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.overrides import handle_torch_function, has_torch_function
from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten
def _svd_meta(A: TensorLikeType, *, full_matrices: bool) -> Tuple[TensorLikeType, TensorLikeType, TensorLikeType]:
    utils.check_is_matrix(A, 'linalg.svd')
    utils.check_fp_or_complex(A.dtype, 'linalg.svd', allow_low_precision_dtypes=False)
    A_shape = A.shape
    batch = A_shape[:-2]
    m, n = A_shape[-2:]
    k = min(m, n)
    shape_U = batch + (m, m if full_matrices else k)
    strides_U = utils.make_contiguous_strides_for(shape_U, row_major=False)
    U = TensorMeta(shape=shape_U, strides=strides_U, dtype=A.dtype, device=A.device)
    shape_S = batch + (k,)
    strides_S = utils.make_contiguous_strides_for(shape_S)
    S = TensorMeta(shape=shape_S, strides=strides_S, dtype=utils.corresponding_real_dtype(A.dtype) if A.is_complex() else A.dtype, device=A.device)
    shape_Vh = batch + (n if full_matrices else k, n)
    is_cuda = A.device.type == 'cuda'
    strides_Vh = utils.make_contiguous_strides_for(shape_Vh, row_major=is_cuda)
    Vh = TensorMeta(shape=shape_Vh, strides=strides_Vh, dtype=A.dtype, device=A.device)
    if A.numel() != 0 and Vh.is_complex() and torch.cuda.is_available():
        Vh = Vh.conj()
    return (U, S, Vh)
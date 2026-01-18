import builtins
import collections
import inspect
import itertools
import math
import operator
import warnings
from collections.abc import Iterable
from enum import Enum
from functools import partial, reduce, singledispatch, wraps
from typing import Any, Callable, Dict, List, Optional, overload, Sequence, Tuple, Union
import torch
import torch._prims as prims
import torch._prims_common as utils
from torch import sym_float, sym_int
from torch._prims_common import (
from torch._prims_common.wrappers import (
from torch._decomp import register_decomposition
import torch._refs._conversions
import torch._refs.fft
import torch._refs.linalg
import torch._refs.nn.functional
import torch._refs.special
def _reduction(a: TensorLikeType, prim: Callable, *, has_identity: bool=True, accepts_dim_tuple: bool=True, dims: Optional[DimsType]=None, keepdims: bool=False, dtype: Optional[torch.dtype]=None, out: Optional[Tensor]=None, output_dtype_kind: REDUCTION_OUTPUT_TYPE_KIND) -> TensorLikeType:
    assert isinstance(a, TensorLike)
    if a.ndim > 64:
        raise RuntimeError(f'Received a tensor with {a.ndim} dimensions, but only tensors with up to 64 dims are supported!')
    if out is not None:
        assert isinstance(out, TensorLike)
        if dtype is not None:
            if dtype != out.dtype:
                raise RuntimeError('dtype argument and out dtype must match in reduction')
    if not accepts_dim_tuple:
        assert dims is None or isinstance(dims, Dim)
    if isinstance(dims, Dim):
        dims = (dims,)
    dims = utils.reduction_dims(a.shape, dims)
    if not has_identity:
        valid_shape = a.ndim == 0 or py_all((a.shape[i] for i in dims))
        if not valid_shape:
            raise RuntimeError('reducing over zero-size dimension for reduction operation without identity')
    computation_dtype, result_dtype = utils.reduction_dtypes(a, output_dtype_kind, dtype)
    a = _maybe_convert_to_dtype(a, computation_dtype)
    result = prim(a, dims)
    if keepdims:
        output_shape = [a.shape[i] if i not in dims else 1 for i in range(a.ndim)]
        broadcast_dims = [i for i in range(a.ndim) if i not in dims]
        result = prims.broadcast_in_dim(result, output_shape, broadcast_dims)
    if out is not None:
        assert result_dtype is not None
        if dtype is not None and result_dtype != out.dtype:
            raise RuntimeError('Expected the dtype of reduction result and out to match')
        out = _maybe_resize_out(out, result.shape)
        return _safe_copy_out(copy_from=result, copy_to=out)
    if result.dtype != result_dtype and result_dtype is not None:
        result = prims.convert_element_type(result, result_dtype)
    return result
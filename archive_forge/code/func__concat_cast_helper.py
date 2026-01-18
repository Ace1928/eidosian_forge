from __future__ import annotations
import builtins
import itertools
import operator
from typing import Optional, Sequence
import torch
from . import _dtypes_impl, _util
from ._normalizations import (
from torch import broadcast_shapes
def _concat_cast_helper(tensors, out=None, dtype=None, casting='same_kind'):
    """Figure out dtypes, cast if necessary."""
    if out is not None or dtype is not None:
        out_dtype = out.dtype.torch_dtype if dtype is None else dtype
    else:
        out_dtype = _dtypes_impl.result_type_impl(*tensors)
    tensors = _util.typecast_tensors(tensors, out_dtype, casting)
    return tensors
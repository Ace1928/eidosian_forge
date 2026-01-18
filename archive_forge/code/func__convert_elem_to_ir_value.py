from __future__ import annotations  # remove after python 3.11
from functools import wraps
from typing import List, Optional, Sequence, Tuple, TypeVar
from .._C.libtriton.triton import ir
from ..common.build import is_hip
from . import core as tl
def _convert_elem_to_ir_value(builder, elem, require_i64):
    if isinstance(elem, int):
        elem = tl.constexpr(elem)
    if isinstance(elem, tl.constexpr):
        return builder.get_int64(elem.value) if require_i64 else builder.get_int32(elem.value)
    elif isinstance(elem, tl.tensor):
        assert elem.numel.value == 1, 'Expected a scalar in shape/strides/offsets'
        assert elem.dtype.is_int(), 'Expected an integer scalar type in shape/strides/offsets'
        if elem.dtype != tl.int64 and require_i64:
            return builder.create_int_cast(elem.handle, builder.get_int64_ty(), elem.dtype.is_int_signed())
        elif elem.dtype != tl.int32:
            return builder.create_int_cast(elem.handle, builder.get_int32_ty(), elem.dtype.is_int_signed())
        return elem.handle
    assert False, f'Unsupported element type in shape/strides/offsets: {type(elem)}'
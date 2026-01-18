from __future__ import annotations  # remove after python 3.11
from functools import wraps
from typing import List, Optional, Sequence, Tuple, TypeVar
from .._C.libtriton.triton import ir
from ..common.build import is_hip
from . import core as tl
def _convert_to_ir_values(builder, list_like, require_i64=True):
    if hasattr(list_like, '__iter__'):
        return [_convert_elem_to_ir_value(builder, elem, require_i64) for elem in list_like]
    return [_convert_elem_to_ir_value(builder, list_like, require_i64)]
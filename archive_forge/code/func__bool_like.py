from __future__ import annotations  # remove after python 3.11
from functools import wraps
from typing import List, Optional, Sequence, Tuple, TypeVar
from .._C.libtriton.triton import ir
from ..common.build import is_hip
from . import core as tl
def _bool_like(v: tl.tensor) -> tl.block_type:
    if not v.type.is_block():
        return tl.int1
    shape = v.type.shape
    return tl.block_type(tl.int1, shape)
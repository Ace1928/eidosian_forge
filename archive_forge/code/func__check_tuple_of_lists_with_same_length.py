from __future__ import annotations
from typing import TYPE_CHECKING, Any, cast
import numpy as np
from contourpy import FillType, LineType
from contourpy.enum_util import as_fill_type, as_line_type
from contourpy.types import MOVETO, code_dtype, offset_dtype, point_dtype
def _check_tuple_of_lists_with_same_length(maybe_tuple: Any, tuple_length: int, allow_empty_lists: bool=True) -> None:
    if not isinstance(maybe_tuple, tuple):
        raise TypeError(f'Expected tuple not {type(maybe_tuple)}')
    if len(maybe_tuple) != tuple_length:
        raise ValueError(f'Expected tuple of length {tuple_length} not {len(maybe_tuple)}')
    for maybe_list in maybe_tuple:
        if not isinstance(maybe_list, list):
            msg = f'Expected tuple to contain {tuple_length} lists but found a {type(maybe_list)}'
            raise TypeError(msg)
    lengths = [len(item) for item in maybe_tuple]
    if len(set(lengths)) != 1:
        msg = f'Expected {tuple_length} lists with same length but lengths are {lengths}'
        raise ValueError(msg)
    if not allow_empty_lists and lengths[0] == 0:
        raise ValueError(f'Expected {tuple_length} non-empty lists')
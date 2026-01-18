from __future__ import annotations  # remove after python 3.11
from functools import wraps
from typing import List, Optional, Sequence, Tuple, TypeVar
from .._C.libtriton.triton import ir
from ..common.build import is_hip
from . import core as tl
def _str_to_padding_option(padding_option):
    padding = None
    if padding_option:
        if padding_option == 'zero':
            padding = ir.PADDING_OPTION.PAD_ZERO
        elif padding_option == 'nan':
            padding = ir.PADDING_OPTION.PAD_NAN
        else:
            raise ValueError(f'Padding option {padding_option} not supported')
    return padding
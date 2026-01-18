import math
import numbers
import os
import cupy
from ._util import _get_inttype
def _init_marker(int_dtype):
    """use a minimum value that is appropriate to the integer dtype"""
    if int_dtype == cupy.int16:
        marker = -32768
    elif int_dtype == cupy.int32:
        marker = -2147483648 // 2
    else:
        raise ValueError('expected int_dtype to be either cupy.int16 or cupy.int32')
    return marker
import functools
import numbers
import sys
import numpy as np
from . import numerictypes as _nt
from .umath import absolute, isinf, isfinite, isnat
from . import multiarray
from .multiarray import (array, dragon4_positional, dragon4_scientific,
from .fromnumeric import any
from .numeric import concatenate, asarray, errstate
from .numerictypes import (longlong, intc, int_, float_, complex_, bool_,
from .overrides import array_function_dispatch, set_module
import operator
import warnings
import contextlib
def _array_repr_implementation(arr, max_line_width=None, precision=None, suppress_small=None, array2string=array2string):
    """Internal version of array_repr() that allows overriding array2string."""
    if max_line_width is None:
        max_line_width = _format_options['linewidth']
    if type(arr) is not ndarray:
        class_name = type(arr).__name__
    else:
        class_name = 'array'
    skipdtype = dtype_is_implied(arr.dtype) and arr.size > 0
    prefix = class_name + '('
    suffix = ')' if skipdtype else ','
    if _format_options['legacy'] <= 113 and arr.shape == () and (not arr.dtype.names):
        lst = repr(arr.item())
    elif arr.size > 0 or arr.shape == (0,):
        lst = array2string(arr, max_line_width, precision, suppress_small, ', ', prefix, suffix=suffix)
    else:
        lst = '[], shape=%s' % (repr(arr.shape),)
    arr_str = prefix + lst + suffix
    if skipdtype:
        return arr_str
    dtype_str = 'dtype={})'.format(dtype_short_repr(arr.dtype))
    last_line_len = len(arr_str) - (arr_str.rfind('\n') + 1)
    spacer = ' '
    if _format_options['legacy'] <= 113:
        if issubclass(arr.dtype.type, flexible):
            spacer = '\n' + ' ' * len(class_name + '(')
    elif last_line_len + len(dtype_str) + 1 > max_line_width:
        spacer = '\n' + ' ' * len(class_name + '(')
    return arr_str + spacer + dtype_str
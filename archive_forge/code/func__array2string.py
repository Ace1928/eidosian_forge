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
@_recursive_guard()
def _array2string(a, options, separator=' ', prefix=''):
    data = asarray(a)
    if a.shape == ():
        a = data
    if a.size > options['threshold']:
        summary_insert = '...'
        data = _leading_trailing(data, options['edgeitems'])
    else:
        summary_insert = ''
    format_function = _get_format_function(data, **options)
    next_line_prefix = ' '
    next_line_prefix += ' ' * len(prefix)
    lst = _formatArray(a, format_function, options['linewidth'], next_line_prefix, separator, options['edgeitems'], summary_insert, options['legacy'])
    return lst
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
def _make_options_dict(precision=None, threshold=None, edgeitems=None, linewidth=None, suppress=None, nanstr=None, infstr=None, sign=None, formatter=None, floatmode=None, legacy=None):
    """
    Make a dictionary out of the non-None arguments, plus conversion of
    *legacy* and sanity checks.
    """
    options = {k: v for k, v in locals().items() if v is not None}
    if suppress is not None:
        options['suppress'] = bool(suppress)
    modes = ['fixed', 'unique', 'maxprec', 'maxprec_equal']
    if floatmode not in modes + [None]:
        raise ValueError('floatmode option must be one of ' + ', '.join(('"{}"'.format(m) for m in modes)))
    if sign not in [None, '-', '+', ' ']:
        raise ValueError("sign option must be one of ' ', '+', or '-'")
    if legacy == False:
        options['legacy'] = sys.maxsize
    elif legacy == '1.13':
        options['legacy'] = 113
    elif legacy == '1.21':
        options['legacy'] = 121
    elif legacy is None:
        pass
    else:
        warnings.warn("legacy printing option can currently only be '1.13', '1.21', or `False`", stacklevel=3)
    if threshold is not None:
        if not isinstance(threshold, numbers.Number):
            raise TypeError('threshold must be numeric')
        if np.isnan(threshold):
            raise ValueError('threshold must be non-NAN, try sys.maxsize for untruncated representation')
    if precision is not None:
        try:
            options['precision'] = operator.index(precision)
        except TypeError as e:
            raise TypeError('precision must be an integer') from e
    return options
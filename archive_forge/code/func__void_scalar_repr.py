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
def _void_scalar_repr(x):
    """
    Implements the repr for structured-void scalars. It is called from the
    scalartypes.c.src code, and is placed here because it uses the elementwise
    formatters defined above.
    """
    return StructuredVoidFormat.from_data(array(x), **_format_options)(x)
import builtins
import inspect
import operator
import warnings
import textwrap
import re
from functools import reduce
import numpy as np
import numpy.core.umath as umath
import numpy.core.numerictypes as ntypes
from numpy.core import multiarray as mu
from numpy import ndarray, amax, amin, iscomplexobj, bool_, _NoValue
from numpy import array as narray
from numpy.lib.function_base import angle
from numpy.compat import (
from numpy import expand_dims
from numpy.core.numeric import normalize_axis_tuple
frombuffer = _convert2ma(
fromfunction = _convert2ma(
def _recursive_set_fill_value(fillvalue, dt):
    """
    Create a fill value for a structured dtype.

    Parameters
    ----------
    fillvalue : scalar or array_like
        Scalar or array representing the fill value. If it is of shorter
        length than the number of fields in dt, it will be resized.
    dt : dtype
        The structured dtype for which to create the fill value.

    Returns
    -------
    val : tuple
        A tuple of values corresponding to the structured fill value.

    """
    fillvalue = np.resize(fillvalue, len(dt.names))
    output_value = []
    for fval, name in zip(fillvalue, dt.names):
        cdtype = dt[name]
        if cdtype.subdtype:
            cdtype = cdtype.subdtype[0]
        if cdtype.names is not None:
            output_value.append(tuple(_recursive_set_fill_value(fval, cdtype)))
        else:
            output_value.append(np.array(fval, dtype=cdtype).item())
    return tuple(output_value)
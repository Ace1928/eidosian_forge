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
class _MaskedUnaryOperation(_MaskedUFunc):
    """
    Defines masked version of unary operations, where invalid values are
    pre-masked.

    Parameters
    ----------
    mufunc : callable
        The function for which to define a masked version. Made available
        as ``_MaskedUnaryOperation.f``.
    fill : scalar, optional
        Filling value, default is 0.
    domain : class instance
        Domain for the function. Should be one of the ``_Domain*``
        classes. Default is None.

    """

    def __init__(self, mufunc, fill=0, domain=None):
        super().__init__(mufunc)
        self.fill = fill
        self.domain = domain
        ufunc_domain[mufunc] = domain
        ufunc_fills[mufunc] = fill

    def __call__(self, a, *args, **kwargs):
        """
        Execute the call behavior.

        """
        d = getdata(a)
        if self.domain is not None:
            with np.errstate(divide='ignore', invalid='ignore'):
                result = self.f(d, *args, **kwargs)
            m = ~umath.isfinite(result)
            m |= self.domain(d)
            m |= getmask(a)
        else:
            with np.errstate(divide='ignore', invalid='ignore'):
                result = self.f(d, *args, **kwargs)
            m = getmask(a)
        if not result.ndim:
            if m:
                return masked
            return result
        if m is not nomask:
            try:
                np.copyto(result, d, where=m)
            except TypeError:
                pass
        masked_result = result.view(get_masked_subclass(a))
        masked_result._mask = m
        masked_result._update_from(a)
        return masked_result
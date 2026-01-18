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
class _extrema_operation(_MaskedUFunc):
    """
    Generic class for maximum/minimum functions.

    .. note::
      This is the base class for `_maximum_operation` and
      `_minimum_operation`.

    """

    def __init__(self, ufunc, compare, fill_value):
        super().__init__(ufunc)
        self.compare = compare
        self.fill_value_func = fill_value

    def __call__(self, a, b):
        """Executes the call behavior."""
        return where(self.compare(a, b), a, b)

    def reduce(self, target, axis=np._NoValue):
        """Reduce target along the given axis."""
        target = narray(target, copy=False, subok=True)
        m = getmask(target)
        if axis is np._NoValue and target.ndim > 1:
            warnings.warn(f'In the future the default for ma.{self.__name__}.reduce will be axis=0, not the current None, to match np.{self.__name__}.reduce. Explicitly pass 0 or None to silence this warning.', MaskedArrayFutureWarning, stacklevel=2)
            axis = None
        if axis is not np._NoValue:
            kwargs = dict(axis=axis)
        else:
            kwargs = dict()
        if m is nomask:
            t = self.f.reduce(target, **kwargs)
        else:
            target = target.filled(self.fill_value_func(target)).view(type(target))
            t = self.f.reduce(target, **kwargs)
            m = umath.logical_and.reduce(m, **kwargs)
            if hasattr(t, '_mask'):
                t._mask = m
            elif m:
                t = masked
        return t

    def outer(self, a, b):
        """Return the function applied to the outer product of a and b."""
        ma = getmask(a)
        mb = getmask(b)
        if ma is nomask and mb is nomask:
            m = nomask
        else:
            ma = getmaskarray(a)
            mb = getmaskarray(b)
            m = logical_or.outer(ma, mb)
        result = self.f.outer(filled(a), filled(b))
        if not isinstance(result, MaskedArray):
            result = result.view(MaskedArray)
        result._mask = m
        return result
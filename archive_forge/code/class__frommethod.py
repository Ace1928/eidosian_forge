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
class _frommethod:
    """
    Define functions from existing MaskedArray methods.

    Parameters
    ----------
    methodname : str
        Name of the method to transform.

    """

    def __init__(self, methodname, reversed=False):
        self.__name__ = methodname
        self.__doc__ = self.getdoc()
        self.reversed = reversed

    def getdoc(self):
        """Return the doc of the function (from the doc of the method)."""
        meth = getattr(MaskedArray, self.__name__, None) or getattr(np, self.__name__, None)
        signature = self.__name__ + get_object_signature(meth)
        if meth is not None:
            doc = '    %s\n%s' % (signature, getattr(meth, '__doc__', None))
            return doc

    def __call__(self, a, *args, **params):
        if self.reversed:
            args = list(args)
            a, args[0] = (args[0], a)
        marr = asanyarray(a)
        method_name = self.__name__
        method = getattr(type(marr), method_name, None)
        if method is None:
            method = getattr(np, method_name)
        return method(marr, *args, **params)
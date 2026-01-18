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
class _convert2ma:
    """
    Convert functions from numpy to numpy.ma.

    Parameters
    ----------
        _methodname : string
            Name of the method to transform.

    """
    __doc__ = None

    def __init__(self, funcname, np_ret, np_ma_ret, params=None):
        self._func = getattr(np, funcname)
        self.__doc__ = self.getdoc(np_ret, np_ma_ret)
        self._extras = params or {}

    def getdoc(self, np_ret, np_ma_ret):
        """Return the doc of the function (from the doc of the method)."""
        doc = getattr(self._func, '__doc__', None)
        sig = get_object_signature(self._func)
        if doc:
            doc = self._replace_return_type(doc, np_ret, np_ma_ret)
            if sig:
                sig = '%s%s\n' % (self._func.__name__, sig)
            doc = sig + doc
        return doc

    def _replace_return_type(self, doc, np_ret, np_ma_ret):
        """
        Replace documentation of ``np`` function's return type.

        Replaces it with the proper type for the ``np.ma`` function.

        Parameters
        ----------
        doc : str
            The documentation of the ``np`` method.
        np_ret : str
            The return type string of the ``np`` method that we want to
            replace. (e.g. "out : ndarray")
        np_ma_ret : str
            The return type string of the ``np.ma`` method.
            (e.g. "out : MaskedArray")
        """
        if np_ret not in doc:
            raise RuntimeError(f'Failed to replace `{np_ret}` with `{np_ma_ret}`. The documentation string for return type, {np_ret}, is not found in the docstring for `np.{self._func.__name__}`. Fix the docstring for `np.{self._func.__name__}` or update the expected string for return type.')
        return doc.replace(np_ret, np_ma_ret)

    def __call__(self, *args, **params):
        _extras = self._extras
        common_params = set(params).intersection(_extras)
        for p in common_params:
            _extras[p] = params.pop(p)
        result = self._func.__call__(*args, **params).view(MaskedArray)
        if 'fill_value' in common_params:
            result.fill_value = _extras.get('fill_value', None)
        if 'hardmask' in common_params:
            result._hardmask = bool(_extras.get('hard_mask', False))
        return result
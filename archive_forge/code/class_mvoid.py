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
class mvoid(MaskedArray):
    """
    Fake a 'void' object to use for masked array with structured dtypes.
    """

    def __new__(self, data, mask=nomask, dtype=None, fill_value=None, hardmask=False, copy=False, subok=True):
        _data = np.array(data, copy=copy, subok=subok, dtype=dtype)
        _data = _data.view(self)
        _data._hardmask = hardmask
        if mask is not nomask:
            if isinstance(mask, np.void):
                _data._mask = mask
            else:
                try:
                    _data._mask = np.void(mask)
                except TypeError:
                    mdtype = make_mask_descr(dtype)
                    _data._mask = np.array(mask, dtype=mdtype)[()]
        if fill_value is not None:
            _data.fill_value = fill_value
        return _data

    @property
    def _data(self):
        return super()._data[()]

    def __getitem__(self, indx):
        """
        Get the index.

        """
        m = self._mask
        if isinstance(m[indx], ndarray):
            return masked_array(data=self._data[indx], mask=m[indx], fill_value=self._fill_value[indx], hard_mask=self._hardmask)
        if m is not nomask and m[indx]:
            return masked
        return self._data[indx]

    def __setitem__(self, indx, value):
        self._data[indx] = value
        if self._hardmask:
            self._mask[indx] |= getattr(value, '_mask', False)
        else:
            self._mask[indx] = getattr(value, '_mask', False)

    def __str__(self):
        m = self._mask
        if m is nomask:
            return str(self._data)
        rdtype = _replace_dtype_fields(self._data.dtype, 'O')
        data_arr = super()._data
        res = data_arr.astype(rdtype)
        _recursive_printoption(res, self._mask, masked_print_option)
        return str(res)
    __repr__ = __str__

    def __iter__(self):
        """Defines an iterator for mvoid"""
        _data, _mask = (self._data, self._mask)
        if _mask is nomask:
            yield from _data
        else:
            for d, m in zip(_data, _mask):
                if m:
                    yield masked
                else:
                    yield d

    def __len__(self):
        return self._data.__len__()

    def filled(self, fill_value=None):
        """
        Return a copy with masked fields filled with a given value.

        Parameters
        ----------
        fill_value : array_like, optional
            The value to use for invalid entries. Can be scalar or
            non-scalar. If latter is the case, the filled array should
            be broadcastable over input array. Default is None, in
            which case the `fill_value` attribute is used instead.

        Returns
        -------
        filled_void
            A `np.void` object

        See Also
        --------
        MaskedArray.filled

        """
        return asarray(self).filled(fill_value)[()]

    def tolist(self):
        """
    Transforms the mvoid object into a tuple.

    Masked fields are replaced by None.

    Returns
    -------
    returned_tuple
        Tuple of fields
        """
        _mask = self._mask
        if _mask is nomask:
            return self._data.tolist()
        result = []
        for d, m in zip(self._data, self._mask):
            if m:
                result.append(None)
            else:
                result.append(d.item())
        return tuple(result)
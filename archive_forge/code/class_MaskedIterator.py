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
class MaskedIterator:
    """
    Flat iterator object to iterate over masked arrays.

    A `MaskedIterator` iterator is returned by ``x.flat`` for any masked array
    `x`. It allows iterating over the array as if it were a 1-D array,
    either in a for-loop or by calling its `next` method.

    Iteration is done in C-contiguous style, with the last index varying the
    fastest. The iterator can also be indexed using basic slicing or
    advanced indexing.

    See Also
    --------
    MaskedArray.flat : Return a flat iterator over an array.
    MaskedArray.flatten : Returns a flattened copy of an array.

    Notes
    -----
    `MaskedIterator` is not exported by the `ma` module. Instead of
    instantiating a `MaskedIterator` directly, use `MaskedArray.flat`.

    Examples
    --------
    >>> x = np.ma.array(arange(6).reshape(2, 3))
    >>> fl = x.flat
    >>> type(fl)
    <class 'numpy.ma.core.MaskedIterator'>
    >>> for item in fl:
    ...     print(item)
    ...
    0
    1
    2
    3
    4
    5

    Extracting more than a single element b indexing the `MaskedIterator`
    returns a masked array:

    >>> fl[2:4]
    masked_array(data = [2 3],
                 mask = False,
           fill_value = 999999)

    """

    def __init__(self, ma):
        self.ma = ma
        self.dataiter = ma._data.flat
        if ma._mask is nomask:
            self.maskiter = None
        else:
            self.maskiter = ma._mask.flat

    def __iter__(self):
        return self

    def __getitem__(self, indx):
        result = self.dataiter.__getitem__(indx).view(type(self.ma))
        if self.maskiter is not None:
            _mask = self.maskiter.__getitem__(indx)
            if isinstance(_mask, ndarray):
                _mask.shape = result.shape
                result._mask = _mask
            elif isinstance(_mask, np.void):
                return mvoid(result, mask=_mask, hardmask=self.ma._hardmask)
            elif _mask:
                return masked
        return result

    def __setitem__(self, index, value):
        self.dataiter[index] = getdata(value)
        if self.maskiter is not None:
            self.maskiter[index] = getmaskarray(value)

    def __next__(self):
        """
        Return the next value, or raise StopIteration.

        Examples
        --------
        >>> x = np.ma.array([3, 2], mask=[0, 1])
        >>> fl = x.flat
        >>> next(fl)
        3
        >>> next(fl)
        masked
        >>> next(fl)
        Traceback (most recent call last):
          ...
        StopIteration

        """
        d = next(self.dataiter)
        if self.maskiter is not None:
            m = next(self.maskiter)
            if isinstance(m, np.void):
                return mvoid(d, mask=m, hardmask=self.ma._hardmask)
            elif m:
                return masked
        return d
from __future__ import annotations
import operator
from enum import IntEnum
from ._creation_functions import asarray
from ._dtypes import (
from typing import TYPE_CHECKING, Optional, Tuple, Union, Any, SupportsIndex
import types
import cupy as np
from cupy.cuda import Device as _Device
from cupy.cuda import stream as stream_module
from cupy_backends.cuda.api import runtime
from cupy import array_api
def _validate_index(self, key):
    """
        Validate an index according to the array API.

        The array API specification only requires a subset of indices that are
        supported by NumPy. This function will reject any index that is
        allowed by NumPy but not required by the array API specification. We
        always raise ``IndexError`` on such indices (the spec does not require
        any specific behavior on them, but this makes the NumPy array API
        namespace a minimal implementation of the spec). See
        https://data-apis.org/array-api/latest/API_specification/indexing.html
        for the full list of required indexing behavior

        This function raises IndexError if the index ``key`` is invalid. It
        only raises ``IndexError`` on indices that are not already rejected by
        NumPy, as NumPy will already raise the appropriate error on such
        indices. ``shape`` may be None, in which case, only cases that are
        independent of the array shape are checked.

        The following cases are allowed by NumPy, but not specified by the array
        API specification:

        - Indices to not include an implicit ellipsis at the end. That is,
          every axis of an array must be explicitly indexed or an ellipsis
          included. This behaviour is sometimes referred to as flat indexing.

        - The start and stop of a slice may not be out of bounds. In
          particular, for a slice ``i:j:k`` on an axis of size ``n``, only the
          following are allowed:

          - ``i`` or ``j`` omitted (``None``).
          - ``-n <= i <= max(0, n - 1)``.
          - For ``k > 0`` or ``k`` omitted (``None``), ``-n <= j <= n``.
          - For ``k < 0``, ``-n - 1 <= j <= max(0, n - 1)``.

        - Boolean array indices are not allowed as part of a larger tuple
          index.

        - Integer array indices are not allowed (with the exception of 0-D
          arrays, which are treated the same as scalars).

        Additionally, it should be noted that indices that would return a
        scalar in NumPy will return a 0-D array. Array scalars are not allowed
        in the specification, only 0-D arrays. This is done in the
        ``Array._new`` constructor, not this function.

        """
    _key = key if isinstance(key, tuple) else (key,)
    for i in _key:
        if isinstance(i, bool) or not (isinstance(i, SupportsIndex) or isinstance(i, Array) or isinstance(i, np.ndarray) or isinstance(i, slice) or (i == Ellipsis) or (i is None)):
            raise IndexError(f'Single-axes index {i} has type(i)={type(i)!r}, but only integers, slices (:), ellipsis (...), newaxis (None), zero-dimensional integer arrays and boolean arrays are specified in the Array API.')
    nonexpanding_key = []
    single_axes = []
    n_ellipsis = 0
    key_has_mask = False
    for i in _key:
        if i is not None:
            nonexpanding_key.append(i)
            if isinstance(i, Array) or isinstance(i, np.ndarray):
                if i.dtype in _boolean_dtypes:
                    key_has_mask = True
                single_axes.append(i)
            elif i == Ellipsis:
                n_ellipsis += 1
            else:
                single_axes.append(i)
    n_single_axes = len(single_axes)
    if n_ellipsis > 1:
        return
    elif n_ellipsis == 0:
        if not key_has_mask and n_single_axes < self.ndim:
            raise IndexError(f'self.ndim={self.ndim!r}, but the multi-axes index only specifies {n_single_axes} dimensions. If this was intentional, add a trailing ellipsis (...) which expands into as many slices (:) as necessary - this is what np.ndarray arrays implicitly do, but such flat indexing behaviour is not specified in the Array API.')
    if n_ellipsis == 0:
        indexed_shape = self.shape
    else:
        ellipsis_start = None
        for pos, i in enumerate(nonexpanding_key):
            if not (isinstance(i, Array) or isinstance(i, np.ndarray)):
                if i == Ellipsis:
                    ellipsis_start = pos
                    break
        assert ellipsis_start is not None
        ellipsis_end = self.ndim - (n_single_axes - ellipsis_start)
        indexed_shape = self.shape[:ellipsis_start] + self.shape[ellipsis_end:]
    for i, side in zip(single_axes, indexed_shape):
        if isinstance(i, slice):
            if side == 0:
                f_range = '0 (or None)'
            else:
                f_range = f'between -{side} and {side - 1} (or None)'
            if i.start is not None:
                try:
                    start = operator.index(i.start)
                except TypeError:
                    pass
                else:
                    if not -side <= start <= side:
                        raise IndexError(f'Slice {i} contains start={start!r}, but should be {f_range} for an axis of size {side} (out-of-bounds starts are not specified in the Array API)')
            if i.stop is not None:
                try:
                    stop = operator.index(i.stop)
                except TypeError:
                    pass
                else:
                    if not -side <= stop <= side:
                        raise IndexError(f'Slice {i} contains stop={stop!r}, but should be {f_range} for an axis of size {side} (out-of-bounds stops are not specified in the Array API)')
        elif isinstance(i, Array):
            if i.dtype in _boolean_dtypes and len(_key) != 1:
                assert isinstance(key, tuple)
                raise IndexError(f'Single-axes index {i} is a boolean array and len(key)={len(key)!r}, but masking is only specified in the Array API when the array is the sole index.')
            elif i.dtype in _integer_dtypes and i.ndim != 0:
                raise IndexError(f'Single-axes index {i} is a non-zero-dimensional integer array, but advanced integer indexing is not specified in the Array API.')
        elif isinstance(i, tuple):
            raise IndexError(f'Single-axes index {i} is a tuple, but nested tuple indices are not specified in the Array API.')
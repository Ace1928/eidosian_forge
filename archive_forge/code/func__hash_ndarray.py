from __future__ import annotations
import itertools
from typing import TYPE_CHECKING
import numpy as np
from pandas._libs.hashing import hash_object_array
from pandas.core.dtypes.common import is_list_like
from pandas.core.dtypes.dtypes import CategoricalDtype
from pandas.core.dtypes.generic import (
def _hash_ndarray(vals: np.ndarray, encoding: str='utf8', hash_key: str=_default_hash_key, categorize: bool=True) -> npt.NDArray[np.uint64]:
    """
    See hash_array.__doc__.
    """
    dtype = vals.dtype
    if np.issubdtype(dtype, np.complex128):
        hash_real = _hash_ndarray(vals.real, encoding, hash_key, categorize)
        hash_imag = _hash_ndarray(vals.imag, encoding, hash_key, categorize)
        return hash_real + 23 * hash_imag
    if dtype == bool:
        vals = vals.astype('u8')
    elif issubclass(dtype.type, (np.datetime64, np.timedelta64)):
        vals = vals.view('i8').astype('u8', copy=False)
    elif issubclass(dtype.type, np.number) and dtype.itemsize <= 8:
        vals = vals.view(f'u{vals.dtype.itemsize}').astype('u8')
    else:
        if categorize:
            from pandas import Categorical, Index, factorize
            codes, categories = factorize(vals, sort=False)
            dtype = CategoricalDtype(categories=Index(categories), ordered=False)
            cat = Categorical._simple_new(codes, dtype)
            return cat._hash_pandas_object(encoding=encoding, hash_key=hash_key, categorize=False)
        try:
            vals = hash_object_array(vals, hash_key, encoding)
        except TypeError:
            vals = hash_object_array(vals.astype(str).astype(object), hash_key, encoding)
    vals ^= vals >> 30
    vals *= np.uint64(13787848793156543929)
    vals ^= vals >> 27
    vals *= np.uint64(10723151780598845931)
    vals ^= vals >> 31
    return vals
import sys
import abc
import io
import copyreg
import pickle
from numba import cloudpickle
from llvmlite import ir
def _numba_unpickle(address, bytedata, hashed):
    """Used by `numba_unpickle` from _helperlib.c

    Parameters
    ----------
    address : int
    bytedata : bytes
    hashed : bytes

    Returns
    -------
    obj : object
        unpickled object
    """
    key = (address, hashed)
    try:
        obj = _unpickled_memo[key]
    except KeyError:
        _unpickled_memo[key] = obj = cloudpickle.loads(bytedata)
    return obj
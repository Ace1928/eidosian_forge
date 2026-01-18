import sys
import abc
import io
import copyreg
import pickle
from numba import cloudpickle
from llvmlite import ir
def _pickle__CustomPickled(cp):
    """standard pickling for `_CustomPickled`.

    Uses `NumbaPickler` to dump.
    """
    serialized = dumps((cp.ctor, cp.states))
    return (_unpickle__CustomPickled, (serialized,))
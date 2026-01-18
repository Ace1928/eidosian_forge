import operator
from functools import reduce
from mmap import mmap
from numbers import Integral
import numpy as np
class _NullLock:
    """Can be used as no-function dummy object in place of ``threading.lock``.

    The ``_NullLock`` is an object which can be used in place of a
    ``threading.Lock`` object, but doesn't actually do anything.

    It is used by the ``read_segments`` function in the event that a
    ``Lock`` is not provided by the caller.
    """

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False
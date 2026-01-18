import numpy as np
from numba.core.typing.typeof import typeof
from numba.core.typing.asnumbatype import as_numba_type
class prange(object):
    """ Provides a 1D parallel iterator that generates a sequence of integers.
    In non-parallel contexts, prange is identical to range.
    """

    def __new__(cls, *args):
        return range(*args)
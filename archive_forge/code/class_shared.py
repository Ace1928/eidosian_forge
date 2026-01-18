import numpy as np
from collections import defaultdict
import functools
import itertools
from inspect import Signature, Parameter
class shared(Stub):
    """
    Shared memory namespace
    """
    _description_ = '<shared>'

    @stub_function
    def array(shape, dtype):
        """
        Allocate a shared array of the given *shape* and *type*. *shape* is
        either an integer or a tuple of integers representing the array's
        dimensions.  *type* is a :ref:`Numba type <numba-types>` of the
        elements needing to be stored in the array.

        The returned array-like object can be read and written to like any
        normal device array (e.g. through indexing).
        """
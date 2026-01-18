import numpy as np
from collections import defaultdict
import functools
import itertools
from inspect import Signature, Parameter
class compare_and_swap(Stub):
    """compare_and_swap(ary, old, val)

        Conditionally assign ``val`` to the first element of an 1D array ``ary``
        if the current value matches ``old``.

        Supported on int32, int64, uint32, uint64 operands only.

        Returns the old value as if it is loaded atomically.
        """
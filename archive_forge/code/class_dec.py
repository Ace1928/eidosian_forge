import numpy as np
from collections import defaultdict
import functools
import itertools
from inspect import Signature, Parameter
class dec(Stub):
    """dec(ary, idx, val)

        Performs::

           ary[idx] = (value if (array[idx] == 0) or
                       (array[idx] > value) else array[idx] - 1)

        Supported on uint32, and uint64 operands only.

        Returns the old value at the index location as if it is loaded
        atomically.
        """
import numpy as np
from collections import defaultdict
import functools
import itertools
from inspect import Signature, Parameter
class inc(Stub):
    """inc(ary, idx, val)

        Perform atomic ``ary[idx] += 1`` up to val, then reset to 0. Supported
        on uint32, and uint64 operands only.

        Returns the old value at the index location as if it is loaded
        atomically.
        """
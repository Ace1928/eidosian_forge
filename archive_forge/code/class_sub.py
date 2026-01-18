import numpy as np
from collections import defaultdict
import functools
import itertools
from inspect import Signature, Parameter
class sub(Stub):
    """sub(ary, idx, val)

        Perform atomic ``ary[idx] -= val``. Supported on int32, float32, and
        float64 operands only.

        Returns the old value at the index location as if it is loaded
        atomically.
        """
import numpy as np
from collections import defaultdict
import functools
import itertools
from inspect import Signature, Parameter
class cas(Stub):
    """cas(ary, idx, old, val)

        Conditionally assign ``val`` to the element ``idx`` of an array
        ``ary`` if the current value of ``ary[idx]`` matches ``old``.

        Supported on int32, int64, uint32, uint64 operands only.

        Returns the old value as if it is loaded atomically.
        """
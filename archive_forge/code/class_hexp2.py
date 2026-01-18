import numpy as np
from collections import defaultdict
import functools
import itertools
from inspect import Signature, Parameter
class hexp2(Stub):
    """hexp2(a)

        Calculate exponential base 2 (2 ** a) in round to nearest mode.
        Supported on fp16 operands only.

        Returns the exponential base 2 result.

        """
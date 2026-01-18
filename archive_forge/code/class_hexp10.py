import numpy as np
from collections import defaultdict
import functools
import itertools
from inspect import Signature, Parameter
class hexp10(Stub):
    """hexp10(a)

        Calculate exponential base 10 (10 ** a) in round to nearest mode.
        Supported on fp16 operands only.

        Returns the exponential base 10 result.

        """
import numpy as np
from collections import defaultdict
import functools
import itertools
from inspect import Signature, Parameter
class hrsqrt(Stub):
    """hrsqrt(a)

        Calculate the reciprocal square root of the input argument in round
        to nearest even mode. Supported on fp16 operands only.

        Returns the reciprocal square root result.

        """
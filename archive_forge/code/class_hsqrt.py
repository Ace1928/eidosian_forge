import numpy as np
from collections import defaultdict
import functools
import itertools
from inspect import Signature, Parameter
class hsqrt(Stub):
    """hsqrt(a)

        Calculate the square root of the input argument in round to nearest
        mode. Supported on fp16 operands only.

        Returns the square root result.

        """
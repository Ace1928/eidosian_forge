import numpy as np
from collections import defaultdict
import functools
import itertools
from inspect import Signature, Parameter
class hexp(Stub):
    """hexp(a)

        Calculate natural exponential, exp(a), in round to nearest mode.
        Supported on fp16 operands only.

        Returns the natural exponential result.

        """
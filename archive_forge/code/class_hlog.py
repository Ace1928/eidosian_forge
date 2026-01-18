import numpy as np
from collections import defaultdict
import functools
import itertools
from inspect import Signature, Parameter
class hlog(Stub):
    """hlog(a)

        Calculate natural logarithm in round to nearest even mode. Supported
        on fp16 operands only.

        Returns the natural logarithm result.

        """
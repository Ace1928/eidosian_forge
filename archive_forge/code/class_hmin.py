import numpy as np
from collections import defaultdict
import functools
import itertools
from inspect import Signature, Parameter
class hmin(Stub):
    """hmin(a, b)

        Perform fp16 minimum operation, min(a,b). Supported
        on fp16 operands only.

        Returns a if a is less than b, returns b otherwise.

        """
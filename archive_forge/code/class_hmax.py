import numpy as np
from collections import defaultdict
import functools
import itertools
from inspect import Signature, Parameter
class hmax(Stub):
    """hmax(a, b)

        Perform fp16 maximum operation, max(a,b) Supported
        on fp16 operands only.

        Returns a if a is greater than b, returns b otherwise.

        """
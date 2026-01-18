import numpy as np
from collections import defaultdict
import functools
import itertools
from inspect import Signature, Parameter
class hsub(Stub):
    """hsub(a, b)

        Perform fp16 subtraction, (a - b) in round to nearest mode. Supported
        on fp16 operands only.

        Returns the fp16 result of the subtraction.

        """
import numpy as np
from collections import defaultdict
import functools
import itertools
from inspect import Signature, Parameter
class hrint(Stub):
    """hrint(a)

        Round the input argument to nearest integer value. Supported on fp16
        operands only.

        Returns the rounded result.

        """
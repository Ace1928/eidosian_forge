import numpy as np
from collections import defaultdict
import functools
import itertools
from inspect import Signature, Parameter
class hneg(Stub):
    """hneg(a)

        Perform fp16 negation, -(a). Supported on fp16 operands only.

        Returns the fp16 result of the negation.

        """
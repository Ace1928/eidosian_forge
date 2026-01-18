import numpy as np
from collections import defaultdict
import functools
import itertools
from inspect import Signature, Parameter
class heq(Stub):
    """heq(a, b)

        Perform fp16 comparison, (a == b). Supported
        on fp16 operands only.

        Returns True if a and b are equal and False otherwise.

        """
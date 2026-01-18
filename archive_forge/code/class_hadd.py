import numpy as np
from collections import defaultdict
import functools
import itertools
from inspect import Signature, Parameter
class hadd(Stub):
    """hadd(a, b)

        Perform fp16 addition, (a + b) in round to nearest mode. Supported
        on fp16 operands only.

        Returns the fp16 result of the addition.

        """
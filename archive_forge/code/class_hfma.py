import numpy as np
from collections import defaultdict
import functools
import itertools
from inspect import Signature, Parameter
class hfma(Stub):
    """hfma(a, b, c)

        Perform fp16 multiply and accumulate, (a * b) + c in round to nearest
        mode. Supported on fp16 operands only.

        Returns the fp16 result of the multiplication.

        """
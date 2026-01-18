import numpy as np
from collections import defaultdict
import functools
import itertools
from inspect import Signature, Parameter
class hsin(Stub):
    """hsin(a)

        Calculate sine in round to nearest even mode. Supported on fp16
        operands only.

        Returns the sine result.

        """
import numpy as np
from collections import defaultdict
import functools
import itertools
from inspect import Signature, Parameter
class laneid(Stub):
    """
    This thread's lane within a warp. Ranges from 0 to
    :attr:`numba.cuda.warpsize` - 1.
    """
    _description_ = '<laneid>'
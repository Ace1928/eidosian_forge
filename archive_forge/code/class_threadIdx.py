import numpy as np
from collections import defaultdict
import functools
import itertools
from inspect import Signature, Parameter
class threadIdx(Dim3):
    """
    The thread indices in the current thread block. Each index is an integer
    spanning the range from 0 inclusive to the corresponding value of the
    attribute in :attr:`numba.cuda.blockDim` exclusive.
    """
    _description_ = '<threadIdx.{x,y,z}>'
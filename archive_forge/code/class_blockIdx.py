import numpy as np
from collections import defaultdict
import functools
import itertools
from inspect import Signature, Parameter
class blockIdx(Dim3):
    """
    The block indices in the grid of thread blocks. Each index is an integer
    spanning the range from 0 inclusive to the corresponding value of the
    attribute in :attr:`numba.cuda.gridDim` exclusive.
    """
    _description_ = '<blockIdx.{x,y,z}>'
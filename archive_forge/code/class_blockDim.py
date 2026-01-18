import numpy as np
from collections import defaultdict
import functools
import itertools
from inspect import Signature, Parameter
class blockDim(Dim3):
    """
    The shape of a block of threads, as declared when instantiating the kernel.
    This value is the same for all threads in a given kernel launch, even if
    they belong to different blocks (i.e. each block is "full").
    """
    _description_ = '<blockDim.{x,y,z}>'
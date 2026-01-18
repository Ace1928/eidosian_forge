import numpy as np
from collections import defaultdict
import functools
import itertools
from inspect import Signature, Parameter
class brev(Stub):
    """
    brev(x)

    Returns the reverse of the bit pattern of x. For example, 0b10110110
    becomes 0b01101101.
    """
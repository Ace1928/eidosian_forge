import numpy as np
from collections import defaultdict
import functools
import itertools
from inspect import Signature, Parameter
class threadfence_block(Stub):
    """
    A memory fence at thread block level
    """
    _description_ = '<threadfence_block()>'
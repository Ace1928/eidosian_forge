import numpy as np
from collections import defaultdict
import functools
import itertools
from inspect import Signature, Parameter
class threadfence(Stub):
    """
    A memory fence at device level
    """
    _description_ = '<threadfence()>'
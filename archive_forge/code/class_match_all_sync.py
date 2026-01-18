import numpy as np
from collections import defaultdict
import functools
import itertools
from inspect import Signature, Parameter
class match_all_sync(Stub):
    """
    match_all_sync(mask, value)

    Nvvm intrinsic for performing a compare and broadcast across a warp.
    Returns a tuple of (mask, pred), where mask is a mask of threads that have
    same value as the given value from within the masked warp, if they
    all have the same value, otherwise it is 0. Pred is a boolean of whether
    or not all threads in the mask warp have the same warp.
    """
    _description_ = '<match_all_sync()>'
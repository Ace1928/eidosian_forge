import types
import math
from torch import inf
from functools import wraps, partial
import warnings
import weakref
from collections import Counter
from bisect import bisect_right
from .optimizer import Optimizer
@staticmethod
def _annealing_linear(start, end, pct):
    """Linearly anneal from `start` to `end` as pct goes from 0.0 to 1.0."""
    return (end - start) * pct + start
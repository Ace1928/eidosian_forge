import types
import math
from torch import inf
from functools import wraps, partial
import warnings
import weakref
from collections import Counter
from bisect import bisect_right
from .optimizer import Optimizer
class _LRScheduler(LRScheduler):
    pass
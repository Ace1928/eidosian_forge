import types
import math
from torch import inf
from functools import wraps, partial
import warnings
import weakref
from collections import Counter
from bisect import bisect_right
from .optimizer import Optimizer
def _initial_step(self):
    """Initialize step counts and performs a step"""
    self.optimizer._step_count = 0
    self._step_count = 0
    self.step()
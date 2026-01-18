import types
import math
from torch import inf
from functools import wraps, partial
import warnings
import weakref
from collections import Counter
from bisect import bisect_right
from .optimizer import Optimizer
def _get_closed_form_lr(self):
    return [self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2 for base_lr in self.base_lrs]
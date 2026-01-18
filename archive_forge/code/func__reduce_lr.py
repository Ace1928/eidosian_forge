import types
import math
from torch import inf
from functools import wraps, partial
import warnings
import weakref
from collections import Counter
from bisect import bisect_right
from .optimizer import Optimizer
def _reduce_lr(self, epoch):
    for i, param_group in enumerate(self.optimizer.param_groups):
        old_lr = float(param_group['lr'])
        new_lr = max(old_lr * self.factor, self.min_lrs[i])
        if old_lr - new_lr > self.eps:
            param_group['lr'] = new_lr
import torch
from functools import reduce
from .optimizer import Optimizer
def _directional_evaluate(self, closure, x, t, d):
    self._add_grad(t, d)
    loss = float(closure())
    flat_grad = self._gather_flat_grad()
    self._set_param(x)
    return (loss, flat_grad)
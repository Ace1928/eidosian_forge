import torch
from functools import reduce
from .optimizer import Optimizer
def _clone_param(self):
    return [p.clone(memory_format=torch.contiguous_format) for p in self._params]
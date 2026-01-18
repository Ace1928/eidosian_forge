import torch
from functools import reduce
from .optimizer import Optimizer
def _set_param(self, params_data):
    for p, pdata in zip(self._params, params_data):
        p.copy_(pdata)
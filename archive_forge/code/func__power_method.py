from enum import Enum, auto
import torch
from torch import Tensor
from ..utils import parametrize
from ..modules import Module
from .. import functional as F
from typing import Optional
@torch.autograd.no_grad()
def _power_method(self, weight_mat: torch.Tensor, n_power_iterations: int) -> None:
    assert weight_mat.ndim > 1
    for _ in range(n_power_iterations):
        self._u = F.normalize(torch.mv(weight_mat, self._v), dim=0, eps=self.eps, out=self._u)
        self._v = F.normalize(torch.mv(weight_mat.t(), self._u), dim=0, eps=self.eps, out=self._v)
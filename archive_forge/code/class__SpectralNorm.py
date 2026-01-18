from enum import Enum, auto
import torch
from torch import Tensor
from ..utils import parametrize
from ..modules import Module
from .. import functional as F
from typing import Optional
class _SpectralNorm(Module):

    def __init__(self, weight: torch.Tensor, n_power_iterations: int=1, dim: int=0, eps: float=1e-12) -> None:
        super().__init__()
        ndim = weight.ndim
        if dim >= ndim or dim < -ndim:
            raise IndexError(f'Dimension out of range (expected to be in range of [-{ndim}, {ndim - 1}] but got {dim})')
        if n_power_iterations <= 0:
            raise ValueError(f'Expected n_power_iterations to be positive, but got n_power_iterations={n_power_iterations}')
        self.dim = dim if dim >= 0 else dim + ndim
        self.eps = eps
        if ndim > 1:
            self.n_power_iterations = n_power_iterations
            weight_mat = self._reshape_weight_to_matrix(weight)
            h, w = weight_mat.size()
            u = weight_mat.new_empty(h).normal_(0, 1)
            v = weight_mat.new_empty(w).normal_(0, 1)
            self.register_buffer('_u', F.normalize(u, dim=0, eps=self.eps))
            self.register_buffer('_v', F.normalize(v, dim=0, eps=self.eps))
            self._power_method(weight_mat, 15)

    def _reshape_weight_to_matrix(self, weight: torch.Tensor) -> torch.Tensor:
        assert weight.ndim > 1
        if self.dim != 0:
            weight = weight.permute(self.dim, *(d for d in range(weight.dim()) if d != self.dim))
        return weight.flatten(1)

    @torch.autograd.no_grad()
    def _power_method(self, weight_mat: torch.Tensor, n_power_iterations: int) -> None:
        assert weight_mat.ndim > 1
        for _ in range(n_power_iterations):
            self._u = F.normalize(torch.mv(weight_mat, self._v), dim=0, eps=self.eps, out=self._u)
            self._v = F.normalize(torch.mv(weight_mat.t(), self._u), dim=0, eps=self.eps, out=self._v)

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        if weight.ndim == 1:
            return F.normalize(weight, dim=0, eps=self.eps)
        else:
            weight_mat = self._reshape_weight_to_matrix(weight)
            if self.training:
                self._power_method(weight_mat, self.n_power_iterations)
            u = self._u.clone(memory_format=torch.contiguous_format)
            v = self._v.clone(memory_format=torch.contiguous_format)
            sigma = torch.dot(u, torch.mv(weight_mat, v))
            return weight / sigma

    def right_inverse(self, value: torch.Tensor) -> torch.Tensor:
        return value
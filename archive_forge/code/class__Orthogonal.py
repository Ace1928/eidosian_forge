from enum import Enum, auto
import torch
from torch import Tensor
from ..utils import parametrize
from ..modules import Module
from .. import functional as F
from typing import Optional
class _Orthogonal(Module):
    base: Tensor

    def __init__(self, weight, orthogonal_map: _OrthMaps, *, use_trivialization=True) -> None:
        super().__init__()
        if weight.is_complex() and orthogonal_map == _OrthMaps.householder:
            raise ValueError('The householder parametrization does not support complex tensors.')
        self.shape = weight.shape
        self.orthogonal_map = orthogonal_map
        if use_trivialization:
            self.register_buffer('base', None)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        n, k = (X.size(-2), X.size(-1))
        transposed = n < k
        if transposed:
            X = X.mT
            n, k = (k, n)
        if self.orthogonal_map == _OrthMaps.matrix_exp or self.orthogonal_map == _OrthMaps.cayley:
            X = X.tril()
            if n != k:
                X = torch.cat([X, X.new_zeros(n, n - k).expand(*X.shape[:-2], -1, -1)], dim=-1)
            A = X - X.mH
            if self.orthogonal_map == _OrthMaps.matrix_exp:
                Q = torch.matrix_exp(A)
            elif self.orthogonal_map == _OrthMaps.cayley:
                Id = torch.eye(n, dtype=A.dtype, device=A.device)
                Q = torch.linalg.solve(torch.add(Id, A, alpha=-0.5), torch.add(Id, A, alpha=0.5))
            if n != k:
                Q = Q[..., :k]
        else:
            A = X.tril(diagonal=-1)
            tau = 2.0 / (1.0 + (A * A).sum(dim=-2))
            Q = torch.linalg.householder_product(A, tau)
            Q = Q * X.diagonal(dim1=-2, dim2=-1).int().unsqueeze(-2)
        if hasattr(self, 'base'):
            Q = self.base @ Q
        if transposed:
            Q = Q.mT
        return Q

    @torch.autograd.no_grad()
    def right_inverse(self, Q: torch.Tensor) -> torch.Tensor:
        if Q.shape != self.shape:
            raise ValueError(f'Expected a matrix or batch of matrices of shape {self.shape}. Got a tensor of shape {Q.shape}.')
        Q_init = Q
        n, k = (Q.size(-2), Q.size(-1))
        transpose = n < k
        if transpose:
            Q = Q.mT
            n, k = (k, n)
        if not hasattr(self, 'base'):
            if self.orthogonal_map == _OrthMaps.cayley or self.orthogonal_map == _OrthMaps.matrix_exp:
                raise NotImplementedError('It is not possible to assign to the matrix exponential or the Cayley parametrizations when use_trivialization=False.')
            A, tau = torch.geqrf(Q)
            A.diagonal(dim1=-2, dim2=-1).sign_()
            A.diagonal(dim1=-2, dim2=-1)[tau == 0.0] *= -1
            return A.mT if transpose else A
        else:
            if n == k:
                if not _is_orthogonal(Q):
                    Q = _make_orthogonal(Q)
                else:
                    Q = Q.clone()
            else:
                N = torch.randn(*Q.size()[:-2] + (n, n - k), dtype=Q.dtype, device=Q.device)
                Q = torch.cat([Q, N], dim=-1)
                Q = _make_orthogonal(Q)
            self.base = Q
            neg_Id = torch.zeros_like(Q_init)
            neg_Id.diagonal(dim1=-2, dim2=-1).fill_(-1.0)
            return neg_Id
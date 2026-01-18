import functools
import math
import numbers
import operator
import weakref
from typing import List
import torch
import torch.nn.functional as F
from torch.distributions import constraints
from torch.distributions.utils import (
from torch.nn.functional import pad, softplus
class CorrCholeskyTransform(Transform):
    """
    Transforms an uncontrained real vector :math:`x` with length :math:`D*(D-1)/2` into the
    Cholesky factor of a D-dimension correlation matrix. This Cholesky factor is a lower
    triangular matrix with positive diagonals and unit Euclidean norm for each row.
    The transform is processed as follows:

        1. First we convert x into a lower triangular matrix in row order.
        2. For each row :math:`X_i` of the lower triangular part, we apply a *signed* version of
           class :class:`StickBreakingTransform` to transform :math:`X_i` into a
           unit Euclidean length vector using the following steps:
           - Scales into the interval :math:`(-1, 1)` domain: :math:`r_i = \\tanh(X_i)`.
           - Transforms into an unsigned domain: :math:`z_i = r_i^2`.
           - Applies :math:`s_i = StickBreakingTransform(z_i)`.
           - Transforms back into signed domain: :math:`y_i = sign(r_i) * \\sqrt{s_i}`.
    """
    domain = constraints.real_vector
    codomain = constraints.corr_cholesky
    bijective = True

    def _call(self, x):
        x = torch.tanh(x)
        eps = torch.finfo(x.dtype).eps
        x = x.clamp(min=-1 + eps, max=1 - eps)
        r = vec_to_tril_matrix(x, diag=-1)
        z = r ** 2
        z1m_cumprod_sqrt = (1 - z).sqrt().cumprod(-1)
        r = r + torch.eye(r.shape[-1], dtype=r.dtype, device=r.device)
        y = r * pad(z1m_cumprod_sqrt[..., :-1], [1, 0], value=1)
        return y

    def _inverse(self, y):
        y_cumsum = 1 - torch.cumsum(y * y, dim=-1)
        y_cumsum_shifted = pad(y_cumsum[..., :-1], [1, 0], value=1)
        y_vec = tril_matrix_to_vec(y, diag=-1)
        y_cumsum_vec = tril_matrix_to_vec(y_cumsum_shifted, diag=-1)
        t = y_vec / y_cumsum_vec.sqrt()
        x = (t.log1p() - t.neg().log1p()) / 2
        return x

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        y1m_cumsum = 1 - (y * y).cumsum(dim=-1)
        y1m_cumsum_tril = tril_matrix_to_vec(y1m_cumsum, diag=-2)
        stick_breaking_logdet = 0.5 * y1m_cumsum_tril.log().sum(-1)
        tanh_logdet = -2 * (x + softplus(-2 * x) - math.log(2.0)).sum(dim=-1)
        return stick_breaking_logdet + tanh_logdet

    def forward_shape(self, shape):
        if len(shape) < 1:
            raise ValueError('Too few dimensions on input')
        N = shape[-1]
        D = round((0.25 + 2 * N) ** 0.5 + 0.5)
        if D * (D - 1) // 2 != N:
            raise ValueError('Input is not a flattend lower-diagonal number')
        return shape[:-1] + (D, D)

    def inverse_shape(self, shape):
        if len(shape) < 2:
            raise ValueError('Too few dimensions on input')
        if shape[-2] != shape[-1]:
            raise ValueError('Input is not square')
        D = shape[-1]
        N = D * (D - 1) // 2
        return shape[:-2] + (N,)
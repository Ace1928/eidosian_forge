from enum import Enum, auto
import torch
from torch import Tensor
from ..utils import parametrize
from ..modules import Module
from .. import functional as F
from typing import Optional
def _make_orthogonal(A):
    """Assume that A is a tall matrix.

    Compute the Q factor s.t. A = QR (A may be complex) and diag(R) is real and non-negative.
    """
    X, tau = torch.geqrf(A)
    Q = torch.linalg.householder_product(X, tau)
    Q *= X.diagonal(dim1=-2, dim2=-1).sgn().unsqueeze(-2)
    return Q
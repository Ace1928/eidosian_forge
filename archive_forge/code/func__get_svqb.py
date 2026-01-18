from typing import Dict, Optional, Tuple
import torch
from torch import Tensor
from . import _linalg_utils as _utils
from .overrides import handle_torch_function, has_torch_function
def _get_svqb(self, U: Tensor, drop: bool, tau: float) -> Tensor:
    """Return B-orthonormal U.

        .. note:: When `drop` is `False` then `svqb` is based on the
                  Algorithm 4 from [DuerschPhD2015] that is a slight
                  modification of the corresponding algorithm
                  introduced in [StathopolousWu2002].

        Args:

          U (Tensor) : initial approximation, size is (m, n)
          drop (bool) : when True, drop columns that
                     contribution to the `span([U])` is small.
          tau (float) : positive tolerance

        Returns:

          U (Tensor) : B-orthonormal columns (:math:`U^T B U = I`), size
                       is (m, n1), where `n1 = n` if `drop` is `False,
                       otherwise `n1 <= n`.

        """
    if torch.numel(U) == 0:
        return U
    UBU = _utils.qform(self.B, U)
    d = UBU.diagonal(0, -2, -1)
    nz = torch.where(abs(d) != 0.0)
    assert len(nz) == 1, nz
    if len(nz[0]) < len(d):
        U = U[:, nz[0]]
        if torch.numel(U) == 0:
            return U
        UBU = _utils.qform(self.B, U)
        d = UBU.diagonal(0, -2, -1)
        nz = torch.where(abs(d) != 0.0)
        assert len(nz[0]) == len(d)
    d_col = (d ** (-0.5)).reshape(d.shape[0], 1)
    DUBUD = UBU * d_col * _utils.transpose(d_col)
    E, Z = _utils.symeig(DUBUD)
    t = tau * abs(E).max()
    if drop:
        keep = torch.where(E > t)
        assert len(keep) == 1, keep
        E = E[keep[0]]
        Z = Z[:, keep[0]]
        d_col = d_col[keep[0]]
    else:
        E[torch.where(E < t)[0]] = t
    return torch.matmul(U * _utils.transpose(d_col), Z * E ** (-0.5))
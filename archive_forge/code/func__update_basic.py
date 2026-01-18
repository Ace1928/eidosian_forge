from typing import Dict, Optional, Tuple
import torch
from torch import Tensor
from . import _linalg_utils as _utils
from .overrides import handle_torch_function, has_torch_function
def _update_basic(self):
    """
        Update or initialize iteration variables when `method == "basic"`.
        """
    mm = torch.matmul
    ns = self.ivars['converged_end']
    nc = self.ivars['converged_count']
    n = self.iparams['n']
    largest = self.bparams['largest']
    if self.ivars['istep'] == 0:
        Ri = self._get_rayleigh_ritz_transform(self.X)
        M = _utils.qform(_utils.qform(self.A, self.X), Ri)
        E, Z = _utils.symeig(M, largest)
        self.X[:] = mm(self.X, mm(Ri, Z))
        self.E[:] = E
        np = 0
        self.update_residual()
        nc = self.update_converged_count()
        self.S[..., :n] = self.X
        W = _utils.matmul(self.iK, self.R)
        self.ivars['converged_end'] = ns = n + np + W.shape[-1]
        self.S[:, n + np:ns] = W
    else:
        S_ = self.S[:, nc:ns]
        Ri = self._get_rayleigh_ritz_transform(S_)
        M = _utils.qform(_utils.qform(self.A, S_), Ri)
        E_, Z = _utils.symeig(M, largest)
        self.X[:, nc:] = mm(S_, mm(Ri, Z[:, :n - nc]))
        self.E[nc:] = E_[:n - nc]
        P = mm(S_, mm(Ri, Z[:, n:2 * n - nc]))
        np = P.shape[-1]
        self.update_residual()
        nc = self.update_converged_count()
        self.S[..., :n] = self.X
        self.S[:, n:n + np] = P
        W = _utils.matmul(self.iK, self.R[:, nc:])
        self.ivars['converged_end'] = ns = n + np + W.shape[-1]
        self.S[:, n + np:ns] = W
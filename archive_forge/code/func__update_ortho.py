from typing import Dict, Optional, Tuple
import torch
from torch import Tensor
from . import _linalg_utils as _utils
from .overrides import handle_torch_function, has_torch_function
def _update_ortho(self):
    """
        Update or initialize iteration variables when `method == "ortho"`.
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
        self.X = mm(self.X, mm(Ri, Z))
        self.update_residual()
        np = 0
        nc = self.update_converged_count()
        self.S[:, :n] = self.X
        W = self._get_ortho(self.R, self.X)
        ns = self.ivars['converged_end'] = n + np + W.shape[-1]
        self.S[:, n + np:ns] = W
    else:
        S_ = self.S[:, nc:ns]
        E_, Z = _utils.symeig(_utils.qform(self.A, S_), largest)
        self.X[:, nc:] = mm(S_, Z[:, :n - nc])
        self.E[nc:] = E_[:n - nc]
        P = mm(S_, mm(Z[:, n - nc:], _utils.basis(_utils.transpose(Z[:n - nc, n - nc:]))))
        np = P.shape[-1]
        self.update_residual()
        nc = self.update_converged_count()
        self.S[:, :n] = self.X
        self.S[:, n:n + np] = P
        W = self._get_ortho(self.R[:, nc:], self.S[:, :n + np])
        ns = self.ivars['converged_end'] = n + np + W.shape[-1]
        self.S[:, n + np:ns] = W
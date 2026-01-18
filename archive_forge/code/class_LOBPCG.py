from typing import Dict, Optional, Tuple
import torch
from torch import Tensor
from . import _linalg_utils as _utils
from .overrides import handle_torch_function, has_torch_function
class LOBPCG:
    """Worker class of LOBPCG methods."""

    def __init__(self, A: Optional[Tensor], B: Optional[Tensor], X: Tensor, iK: Optional[Tensor], iparams: Dict[str, int], fparams: Dict[str, float], bparams: Dict[str, bool], method: str, tracker: None) -> None:
        self.A = A
        self.B = B
        self.iK = iK
        self.iparams = iparams
        self.fparams = fparams
        self.bparams = bparams
        self.method = method
        self.tracker = tracker
        m = iparams['m']
        n = iparams['n']
        self.X = X
        self.E = torch.zeros((n,), dtype=X.dtype, device=X.device)
        self.R = torch.zeros((m, n), dtype=X.dtype, device=X.device)
        self.S = torch.zeros((m, 3 * n), dtype=X.dtype, device=X.device)
        self.tvars: Dict[str, Tensor] = {}
        self.ivars: Dict[str, int] = {'istep': 0}
        self.fvars: Dict[str, float] = {'_': 0.0}
        self.bvars: Dict[str, bool] = {'_': False}

    def __str__(self):
        lines = ['LOPBCG:']
        lines += [f'  iparams={self.iparams}']
        lines += [f'  fparams={self.fparams}']
        lines += [f'  bparams={self.bparams}']
        lines += [f'  ivars={self.ivars}']
        lines += [f'  fvars={self.fvars}']
        lines += [f'  bvars={self.bvars}']
        lines += [f'  tvars={self.tvars}']
        lines += [f'  A={self.A}']
        lines += [f'  B={self.B}']
        lines += [f'  iK={self.iK}']
        lines += [f'  X={self.X}']
        lines += [f'  E={self.E}']
        r = ''
        for line in lines:
            r += line + '\n'
        return r

    def update(self):
        """Set and update iteration variables."""
        if self.ivars['istep'] == 0:
            X_norm = float(torch.norm(self.X))
            iX_norm = X_norm ** (-1)
            A_norm = float(torch.norm(_utils.matmul(self.A, self.X))) * iX_norm
            B_norm = float(torch.norm(_utils.matmul(self.B, self.X))) * iX_norm
            self.fvars['X_norm'] = X_norm
            self.fvars['A_norm'] = A_norm
            self.fvars['B_norm'] = B_norm
            self.ivars['iterations_left'] = self.iparams['niter']
            self.ivars['converged_count'] = 0
            self.ivars['converged_end'] = 0
        if self.method == 'ortho':
            self._update_ortho()
        else:
            self._update_basic()
        self.ivars['iterations_left'] = self.ivars['iterations_left'] - 1
        self.ivars['istep'] = self.ivars['istep'] + 1

    def update_residual(self):
        """Update residual R from A, B, X, E."""
        mm = _utils.matmul
        self.R = mm(self.A, self.X) - mm(self.B, self.X) * self.E

    def update_converged_count(self):
        """Determine the number of converged eigenpairs using backward stable
        convergence criterion, see discussion in Sec 4.3 of [DuerschEtal2018].

        Users may redefine this method for custom convergence criteria.
        """
        prev_count = self.ivars['converged_count']
        tol = self.fparams['tol']
        A_norm = self.fvars['A_norm']
        B_norm = self.fvars['B_norm']
        E, X, R = (self.E, self.X, self.R)
        rerr = torch.norm(R, 2, (0,)) * (torch.norm(X, 2, (0,)) * (A_norm + E[:X.shape[-1]] * B_norm)) ** (-1)
        converged = rerr < tol
        count = 0
        for b in converged:
            if not b:
                break
            count += 1
        assert count >= prev_count, f'the number of converged eigenpairs (was {prev_count}, got {count}) cannot decrease'
        self.ivars['converged_count'] = count
        self.tvars['rerr'] = rerr
        return count

    def stop_iteration(self):
        """Return True to stop iterations.

        Note that tracker (if defined) can force-stop iterations by
        setting ``worker.bvars['force_stop'] = True``.
        """
        return self.bvars.get('force_stop', False) or self.ivars['iterations_left'] == 0 or self.ivars['converged_count'] >= self.iparams['k']

    def run(self):
        """Run LOBPCG iterations.

        Use this method as a template for implementing LOBPCG
        iteration scheme with custom tracker that is compatible with
        TorchScript.
        """
        self.update()
        if not torch.jit.is_scripting() and self.tracker is not None:
            self.call_tracker()
        while not self.stop_iteration():
            self.update()
            if not torch.jit.is_scripting() and self.tracker is not None:
                self.call_tracker()

    @torch.jit.unused
    def call_tracker(self):
        """Interface for tracking iteration process in Python mode.

        Tracking the iteration process is disabled in TorchScript
        mode. In fact, one should specify tracker=None when JIT
        compiling functions using lobpcg.
        """
        pass

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

    def _get_rayleigh_ritz_transform(self, S):
        """Return a transformation matrix that is used in Rayleigh-Ritz
        procedure for reducing a general eigenvalue problem :math:`(S^TAS)
        C = (S^TBS) C E` to a standard eigenvalue problem :math: `(Ri^T
        S^TAS Ri) Z = Z E` where `C = Ri Z`.

        .. note:: In the original Rayleight-Ritz procedure in
          [DuerschEtal2018], the problem is formulated as follows::

            SAS = S^T A S
            SBS = S^T B S
            D = (<diagonal matrix of SBS>) ** -1/2
            R^T R = Cholesky(D SBS D)
            Ri = D R^-1
            solve symeig problem Ri^T SAS Ri Z = Theta Z
            C = Ri Z

          To reduce the number of matrix products (denoted by empty
          space between matrices), here we introduce element-wise
          products (denoted by symbol `*`) so that the Rayleight-Ritz
          procedure becomes::

            SAS = S^T A S
            SBS = S^T B S
            d = (<diagonal of SBS>) ** -1/2    # this is 1-d column vector
            dd = d d^T                         # this is 2-d matrix
            R^T R = Cholesky(dd * SBS)
            Ri = R^-1 * d                      # broadcasting
            solve symeig problem Ri^T SAS Ri Z = Theta Z
            C = Ri Z

          where `dd` is 2-d matrix that replaces matrix products `D M
          D` with one element-wise product `M * dd`; and `d` replaces
          matrix product `D M` with element-wise product `M *
          d`. Also, creating the diagonal matrix `D` is avoided.

        Args:
        S (Tensor): the matrix basis for the search subspace, size is
                    :math:`(m, n)`.

        Returns:
        Ri (tensor): upper-triangular transformation matrix of size
                     :math:`(n, n)`.

        """
        B = self.B
        mm = torch.matmul
        SBS = _utils.qform(B, S)
        d_row = SBS.diagonal(0, -2, -1) ** (-0.5)
        d_col = d_row.reshape(d_row.shape[0], 1)
        R = torch.linalg.cholesky(SBS * d_row * d_col, upper=True)
        return torch.linalg.solve_triangular(R, d_row.diag_embed(), upper=True, left=False)

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

    def _get_ortho(self, U, V):
        """Return B-orthonormal U with columns are B-orthogonal to V.

        .. note:: When `bparams["ortho_use_drop"] == False` then
                  `_get_ortho` is based on the Algorithm 3 from
                  [DuerschPhD2015] that is a slight modification of
                  the corresponding algorithm introduced in
                  [StathopolousWu2002]. Otherwise, the method
                  implements Algorithm 6 from [DuerschPhD2015]

        .. note:: If all U columns are B-collinear to V then the
                  returned tensor U will be empty.

        Args:

          U (Tensor) : initial approximation, size is (m, n)
          V (Tensor) : B-orthogonal external basis, size is (m, k)

        Returns:

          U (Tensor) : B-orthonormal columns (:math:`U^T B U = I`)
                       such that :math:`V^T B U=0`, size is (m, n1),
                       where `n1 = n` if `drop` is `False, otherwise
                       `n1 <= n`.
        """
        mm = torch.matmul
        mm_B = _utils.matmul
        m = self.iparams['m']
        tau_ortho = self.fparams['ortho_tol']
        tau_drop = self.fparams['ortho_tol_drop']
        tau_replace = self.fparams['ortho_tol_replace']
        i_max = self.iparams['ortho_i_max']
        j_max = self.iparams['ortho_j_max']
        use_drop = self.bparams['ortho_use_drop']
        for vkey in list(self.fvars.keys()):
            if vkey.startswith('ortho_') and vkey.endswith('_rerr'):
                self.fvars.pop(vkey)
        self.ivars.pop('ortho_i', 0)
        self.ivars.pop('ortho_j', 0)
        BV_norm = torch.norm(mm_B(self.B, V))
        BU = mm_B(self.B, U)
        VBU = mm(_utils.transpose(V), BU)
        i = j = 0
        stats = ''
        for i in range(i_max):
            U = U - mm(V, VBU)
            drop = False
            tau_svqb = tau_drop
            for j in range(j_max):
                if use_drop:
                    U = self._get_svqb(U, drop, tau_svqb)
                    drop = True
                    tau_svqb = tau_replace
                else:
                    U = self._get_svqb(U, False, tau_replace)
                if torch.numel(U) == 0:
                    self.ivars['ortho_i'] = i
                    self.ivars['ortho_j'] = j
                    return U
                BU = mm_B(self.B, U)
                UBU = mm(_utils.transpose(U), BU)
                U_norm = torch.norm(U)
                BU_norm = torch.norm(BU)
                R = UBU - torch.eye(UBU.shape[-1], device=UBU.device, dtype=UBU.dtype)
                R_norm = torch.norm(R)
                rerr = float(R_norm) * float(BU_norm * U_norm) ** (-1)
                vkey = f'ortho_UBUmI_rerr[{i}, {j}]'
                self.fvars[vkey] = rerr
                if rerr < tau_ortho:
                    break
            VBU = mm(_utils.transpose(V), BU)
            VBU_norm = torch.norm(VBU)
            U_norm = torch.norm(U)
            rerr = float(VBU_norm) * float(BV_norm * U_norm) ** (-1)
            vkey = f'ortho_VBU_rerr[{i}]'
            self.fvars[vkey] = rerr
            if rerr < tau_ortho:
                break
            if m < U.shape[-1] + V.shape[-1]:
                B = self.B
                assert B is not None
                raise ValueError(f'Overdetermined shape of U: #B-cols(={B.shape[-1]}) >= #U-cols(={U.shape[-1]}) + #V-cols(={V.shape[-1]}) must hold')
        self.ivars['ortho_i'] = i
        self.ivars['ortho_j'] = j
        return U
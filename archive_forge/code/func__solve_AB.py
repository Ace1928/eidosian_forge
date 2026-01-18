import numpy as np
import numpy.linalg as npl
from numpy.linalg import slogdet
from statsmodels.tools.decorators import deprecated_alias
from statsmodels.tools.numdiff import approx_fprime, approx_hess
import statsmodels.tsa.base.tsa_model as tsbase
from statsmodels.tsa.vector_ar.irf import IRAnalysis
import statsmodels.tsa.vector_ar.util as util
from statsmodels.tsa.vector_ar.var_model import VARProcess, VARResults
def _solve_AB(self, start_params, maxiter, override=False, solver='bfgs'):
    """
        Solves for MLE estimate of structural parameters

        Parameters
        ----------

        override : bool, default False
            If True, returns estimates of A and B without checking
            order or rank condition
        solver : str or None, optional
            Solver to be used. The default is 'nm' (Nelder-Mead). Other
            choices are 'bfgs', 'newton' (Newton-Raphson), 'cg'
            conjugate, 'ncg' (non-conjugate gradient), and 'powell'.
        maxiter : int, optional
            The maximum number of iterations. Default is 500.

        Returns
        -------
        A_solve, B_solve: ML solutions for A, B matrices
        """
    A_mask = self.A_mask
    B_mask = self.B_mask
    A = self.A
    B = self.B
    A_len = len(A[A_mask])
    A[A_mask] = start_params[:A_len]
    B[B_mask] = start_params[A_len:]
    if not override:
        J = self._compute_J(A, B)
        self.check_order(J)
        self.check_rank(J)
    else:
        print('Order/rank conditions have not been checked')
    retvals = super().fit(start_params=start_params, method=solver, maxiter=maxiter, gtol=1e-20, disp=False).params
    A[A_mask] = retvals[:A_len]
    B[B_mask] = retvals[A_len:]
    return (A, B)
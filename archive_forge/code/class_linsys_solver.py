from __future__ import print_function
from builtins import range
from builtins import object
import numpy as np
import scipy as sp
import scipy.sparse as spspa
import scipy.sparse.linalg as spla
import numpy.linalg as la
import time   # Time execution
class linsys_solver(object):
    """
    Linear systems solver
    """

    def __init__(self, work):
        """
        Initialize structure for KKT system solution
        """
        KKT = spspa.vstack([spspa.hstack([work.data.P + work.settings.sigma * spspa.eye(work.data.n), work.data.A.T]), spspa.hstack([work.data.A, -spspa.diags(work.rho_inv_vec)])])
        self.kkt_factor = spla.splu(KKT.tocsc())

    def solve(self, rhs):
        """
        Solve linear system with given factorization
        """
        return self.kkt_factor.solve(rhs)
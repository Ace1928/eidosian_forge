import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse import csc_matrix, issparse, eye
from scipy.sparse.linalg import splu
from scipy.optimize._numdiff import group_columns
from .common import (validate_max_step, validate_tol, select_initial_step,
from .base import OdeSolver, DenseOutput
def _compute_dense_output(self):
    Q = np.dot(self.Z.T, P)
    return RadauDenseOutput(self.t_old, self.t, self.y_old, Q)
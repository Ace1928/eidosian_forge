import scipy.sparse as sps
import numpy as np
from .equality_constrained_sqp import equality_constrained_sqp
from scipy.sparse.linalg import LinearOperator
def _compute_function(self, f, c_ineq, s):
    s[self.enforce_feasibility] = -c_ineq[self.enforce_feasibility]
    log_s = [np.log(s_i) if s_i > 0 else -np.inf for s_i in s]
    return f - self.barrier_parameter * np.sum(log_s)
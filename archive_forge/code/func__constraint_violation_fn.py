import warnings
import numpy as np
from scipy.optimize import OptimizeResult, minimize
from scipy.optimize._optimize import _status_message, _wrap_callback
from scipy._lib._util import check_random_state, MapWrapper, _FunctionWrapper
from scipy.optimize._constraints import (Bounds, new_bounds_to_old,
from scipy.sparse import issparse
def _constraint_violation_fn(self, x):
    """
        Calculates total constraint violation for all the constraints, for a
        set of solutions.

        Parameters
        ----------
        x : ndarray
            Solution vector(s). Has shape (S, N), or (N,), where S is the
            number of solutions to investigate and N is the number of
            parameters.

        Returns
        -------
        cv : ndarray
            Total violation of constraints. Has shape ``(S, M)``, where M is
            the total number of constraint components (which is not necessarily
            equal to len(self._wrapped_constraints)).
        """
    S = np.size(x) // self.parameter_count
    _out = np.zeros((S, self.total_constraints))
    offset = 0
    for con in self._wrapped_constraints:
        c = con.violation(x.T).T
        if c.shape[-1] != con.num_constr or (S > 1 and c.shape[0] != S):
            raise RuntimeError('An array returned from a Constraint has the wrong shape. If `vectorized is False` the Constraint should return an array of shape (M,). If `vectorized is True` then the Constraint must return an array of shape (M, S), where S is the number of solution vectors and M is the number of constraint components in a given Constraint object.')
        c = np.reshape(c, (S, con.num_constr))
        _out[:, offset:offset + con.num_constr] = c
        offset += con.num_constr
    return _out
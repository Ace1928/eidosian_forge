import warnings
import numpy as np
from scipy.optimize import OptimizeResult, minimize
from scipy.optimize._optimize import _status_message, _wrap_callback
from scipy._lib._util import check_random_state, MapWrapper, _FunctionWrapper
from scipy.optimize._constraints import (Bounds, new_bounds_to_old,
from scipy.sparse import issparse
def _accept_trial(self, energy_trial, feasible_trial, cv_trial, energy_orig, feasible_orig, cv_orig):
    """
        Trial is accepted if:
        * it satisfies all constraints and provides a lower or equal objective
          function value, while both the compared solutions are feasible
        - or -
        * it is feasible while the original solution is infeasible,
        - or -
        * it is infeasible, but provides a lower or equal constraint violation
          for all constraint functions.

        This test corresponds to section III of Lampinen [1]_.

        Parameters
        ----------
        energy_trial : float
            Energy of the trial solution
        feasible_trial : float
            Feasibility of trial solution
        cv_trial : array-like
            Excess constraint violation for the trial solution
        energy_orig : float
            Energy of the original solution
        feasible_orig : float
            Feasibility of original solution
        cv_orig : array-like
            Excess constraint violation for the original solution

        Returns
        -------
        accepted : bool

        """
    if feasible_orig and feasible_trial:
        return energy_trial <= energy_orig
    elif feasible_trial and (not feasible_orig):
        return True
    elif not feasible_trial and (cv_trial <= cv_orig).all():
        return True
    return False
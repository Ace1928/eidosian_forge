import warnings
import numpy as np
from scipy.sparse import csc_array, vstack, issparse
from scipy._lib._util import VisibleDeprecationWarning
from ._highs._highs_wrapper import _highs_wrapper  # type: ignore[import]
from ._constraints import LinearConstraint, Bounds
from ._optimize import OptimizeResult
from ._linprog_highs import _highs_to_scipy_status_message
def _constraints_to_components(constraints):
    """
    Convert sequence of constraints to a single set of components A, b_l, b_u.

    `constraints` could be

    1. A LinearConstraint
    2. A tuple representing a LinearConstraint
    3. An invalid object
    4. A sequence of composed entirely of objects of type 1/2
    5. A sequence containing at least one object of type 3

    We want to accept 1, 2, and 4 and reject 3 and 5.
    """
    message = '`constraints` (or each element within `constraints`) must be convertible into an instance of `scipy.optimize.LinearConstraint`.'
    As = []
    b_ls = []
    b_us = []
    if isinstance(constraints, LinearConstraint):
        constraints = [constraints]
    else:
        try:
            iter(constraints)
        except TypeError as exc:
            raise ValueError(message) from exc
        if len(constraints) == 3:
            try:
                constraints = [LinearConstraint(*constraints)]
            except (TypeError, ValueError, VisibleDeprecationWarning):
                pass
    for constraint in constraints:
        if not isinstance(constraint, LinearConstraint):
            try:
                constraint = LinearConstraint(*constraint)
            except TypeError as exc:
                raise ValueError(message) from exc
        As.append(csc_array(constraint.A))
        b_ls.append(np.atleast_1d(constraint.lb).astype(np.float64))
        b_us.append(np.atleast_1d(constraint.ub).astype(np.float64))
    if len(As) > 1:
        A = vstack(As, format='csc')
        b_l = np.concatenate(b_ls)
        b_u = np.concatenate(b_us)
    else:
        A = As[0]
        b_l = b_ls[0]
        b_u = b_us[0]
    return (A, b_l, b_u)
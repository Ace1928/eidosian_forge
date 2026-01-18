import numpy as np
from patsy.util import (have_pandas, atleast_2d_column_default,
from patsy.state import stateful_transform
def _get_crs_dmatrix(x, knots, constraints=None, cyclic=False):
    """Builds a cubic regression spline design matrix.

    Returns design matrix with dimensions len(x) x n
    where:
     - ``n = len(knots) - nrows(constraints)`` for natural CRS
     - ``n = len(knots) - nrows(constraints) - 1`` for cyclic CRS
    for a cubic regression spline smoother

    :param x: The 1-d array values.
    :param knots: The 1-d array knots used for cubic spline parametrization,
     must be sorted in ascending order.
    :param constraints: The 2-d array defining model parameters (``betas``)
     constraints (``np.dot(constraints, betas) = 0``).
    :param cyclic: Indicates whether used cubic regression splines should
     be cyclic or not. Default is ``False``.
    :return: The (2-d array) design matrix.
    """
    dm = _get_free_crs_dmatrix(x, knots, cyclic)
    if constraints is not None:
        dm = _absorb_constraints(dm, constraints)
    return dm
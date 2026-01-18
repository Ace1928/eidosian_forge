import numpy as np
from patsy.util import (have_pandas, atleast_2d_column_default,
from patsy.state import stateful_transform
def _compute_base_functions(x, knots):
    """Computes base functions used for building cubic splines basis.

    .. note:: See 'Generalized Additive Models', Simon N. Wood, 2006, p. 146
      and for the special treatment of ``x`` values outside ``knots`` range
      see 'mgcv' source code, file 'mgcv.c', function 'crspl()', l.249

    :param x: The 1-d array values for which base functions should be computed.
    :param knots: The 1-d array knots used for cubic spline parametrization,
     must be sorted in ascending order.
    :return: 4 arrays corresponding to the 4 base functions ajm, ajp, cjm, cjp
     + the 1-d array of knots lower bounds indices corresponding to
     the given ``x`` values.
    """
    j = _find_knots_lower_bounds(x, knots)
    h = knots[1:] - knots[:-1]
    hj = h[j]
    xj1_x = knots[j + 1] - x
    x_xj = x - knots[j]
    ajm = xj1_x / hj
    ajp = x_xj / hj
    cjm_3 = xj1_x * xj1_x * xj1_x / (6.0 * hj)
    cjm_3[x > np.max(knots)] = 0.0
    cjm_1 = hj * xj1_x / 6.0
    cjm = cjm_3 - cjm_1
    cjp_3 = x_xj * x_xj * x_xj / (6.0 * hj)
    cjp_3[x < np.min(knots)] = 0.0
    cjp_1 = hj * x_xj / 6.0
    cjp = cjp_3 - cjp_1
    return (ajm, ajp, cjm, cjp, j)
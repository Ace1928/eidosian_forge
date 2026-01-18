from collections import namedtuple
import warnings
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import norm
from statsmodels.tools.sm_exceptions import HypothesisTestWarning
def _validate_and_tranform_x_and_y(x, y):
    """Ensure `x` and `y` have proper shape and transform/reshape them if
    required.

    Parameters
    ----------
    x : array_like, 1-D or 2-D
        If `x` is 1-D than it is assumed to be a vector of observations of a
        single random variable. If `x` is 2-D than the rows should be
        observations and the columns are treated as the components of a
        random vector, i.e., each column represents a different component of
        the random vector `x`.
    y : array_like, 1-D or 2-D
        Same as `x`, but only the number of observation has to match that of
        `x`. If `y` is 2-D note that the number of columns of `y` (i.e., the
        number of components in the random vector) does not need to match
        the number of columns in `x`.

    Returns
    -------
    x : array_like, 1-D or 2-D
    y : array_like, 1-D or 2-D

    Raises
    ------
    ValueError
        If `x` and `y` have a different number of observations.

    """
    x = np.asanyarray(x)
    y = np.asanyarray(y)
    if x.shape[0] != y.shape[0]:
        raise ValueError('x and y must have the same number of observations (rows).')
    if len(x.shape) == 1:
        x = x.reshape((x.shape[0], 1))
    if len(y.shape) == 1:
        y = y.reshape((y.shape[0], 1))
    return (x, y)
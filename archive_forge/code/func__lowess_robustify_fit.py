import numpy as np
from numpy.linalg import lstsq
def _lowess_robustify_fit(x_copy, y_copy, fitted, weights, k, n):
    """
    Additional weighted local linear regressions, performed if
    iter>0. They take into account the sizes of the residuals,
    to eliminate the effect of extreme outliers.

    Parameters
    ----------
    x_copy : 1-d ndarray
        The x-values/exogenous part of the data being smoothed
    y_copy : 1-d ndarray
        The y-values/ endogenous part of the data being smoothed
    fitted : 1-d ndarray
        The fitted y-values from the previous iteration
    weights : 2-d ndarray
        An n by k array. The contribution to the weights in the
        local linear fit coming from the distances between the
        x-values
    k : int
        The number of data points which affect the linear fit for
        each estimated point
    n : int
        The total number of points

   Returns
    -------
    Nothing. The fitted values are modified in place.
    """
    nn_indices = [0, k]
    X = np.ones((k, 2))
    residual_weights = np.copy(y_copy)
    residual_weights.shape = (n,)
    residual_weights -= fitted
    residual_weights = np.absolute(residual_weights)
    s = np.median(residual_weights)
    residual_weights /= 6 * s
    too_big = residual_weights >= 1
    _lowess_bisquare(residual_weights)
    residual_weights[too_big] = 0
    for i in range(n):
        total_weights = weights[i, :] * np.sqrt(residual_weights[nn_indices[0]:nn_indices[1]])
        X[:, 1] = x_copy[nn_indices[0]:nn_indices[1]]
        y_i = total_weights * y_copy[nn_indices[0]:nn_indices[1]]
        total_weights.shape = (k, 1)
        beta = lstsq(total_weights * X, y_i, rcond=-1)[0]
        fitted[i] = beta[0] + beta[1] * x_copy[i]
        _lowess_update_nn(x_copy, nn_indices, i + 1)
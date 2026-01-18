import numpy as np
from scipy import stats
def _lm_robust(score, constraint_matrix, score_deriv_inv, cov_score, cov_params=None):
    """general formula for score/LM test

    generalized score or lagrange multiplier test for implicit constraints

    `r(params) = 0`, with gradient `R = d r / d params`

    linear constraints are given by `R params - q = 0`

    It is assumed that all arrays are evaluated at the constrained estimates.


    Parameters
    ----------
    score : ndarray, 1-D
        derivative of objective function at estimated parameters
        of constrained model
    constraint_matrix R : ndarray
        Linear restriction matrix or Jacobian of nonlinear constraints
    score_deriv_inv, Ainv : ndarray, symmetric, square
        inverse of second derivative of objective function
        TODO: could be inverse of OPG or any other estimator if information
        matrix equality holds
    cov_score B :  ndarray, symmetric, square
        covariance matrix of the score. This is the inner part of a sandwich
        estimator.
    cov_params V :  ndarray, symmetric, square
        covariance of full parameter vector evaluated at constrained parameter
        estimate. This can be specified instead of cov_score B.

    Returns
    -------
    lm_stat : float
        score/lagrange multiplier statistic
    p-value : float
        p-value of the LM test based on chisquare distribution

    Notes
    -----

    """
    R, Ainv, B, V = (constraint_matrix, score_deriv_inv, cov_score, cov_params)
    k_constraints = np.linalg.matrix_rank(R)
    tmp = R.dot(Ainv)
    wscore = tmp.dot(score)
    if B is None and V is None:
        lm_stat = score.dot(Ainv.dot(score))
    else:
        if V is None:
            inner = tmp.dot(B).dot(tmp.T)
        else:
            inner = R.dot(V).dot(R.T)
        lm_stat = wscore.dot(np.linalg.solve(inner, wscore))
    pval = stats.chi2.sf(lm_stat, k_constraints)
    return (lm_stat, pval, k_constraints)
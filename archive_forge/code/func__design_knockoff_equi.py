import numpy as np
import pandas as pd
from statsmodels.iolib import summary2
def _design_knockoff_equi(exog):
    """
    Construct an equivariant design matrix for knockoff analysis.

    Follows the 'equi-correlated knockoff approach of equation 2.4 in
    Barber and Candes.

    Constructs a pair of design matrices exogs, exogn such that exogs
    is a scaled/centered version of the input matrix exog, exogn is
    another matrix of the same shape with cov(exogn) = cov(exogs), and
    the covariances between corresponding columns of exogn and exogs
    are as small as possible.
    """
    nobs, nvar = exog.shape
    if nobs < 2 * nvar:
        msg = 'The equivariant knockoff can ony be used when n >= 2*p'
        raise ValueError(msg)
    xnm = np.sum(exog ** 2, 0)
    xnm = np.sqrt(xnm)
    exog = exog / xnm
    xcov = np.dot(exog.T, exog)
    ev, _ = np.linalg.eig(xcov)
    evmin = np.min(ev)
    sl = min(2 * evmin, 1)
    sl = sl * np.ones(nvar)
    exogn = _get_knmat(exog, xcov, sl)
    return (exog, exogn, sl)
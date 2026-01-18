import numpy as np
import pandas as pd
from statsmodels.iolib import summary2
def _get_knmat(exog, xcov, sl):
    nobs, nvar = exog.shape
    ash = np.linalg.inv(xcov)
    ash *= -np.outer(sl, sl)
    i, j = np.diag_indices(nvar)
    ash[i, j] += 2 * sl
    umat = np.random.normal(size=(nobs, nvar))
    u, _ = np.linalg.qr(exog)
    umat -= np.dot(u, np.dot(u.T, umat))
    umat, _ = np.linalg.qr(umat)
    ashr, xc, _ = np.linalg.svd(ash, 0)
    ashr *= np.sqrt(xc)
    ashr = ashr.T
    ex = (sl[:, None] * np.linalg.solve(xcov, exog.T)).T
    exogn = exog - ex + np.dot(umat, ashr)
    return exogn
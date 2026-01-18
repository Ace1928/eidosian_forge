from statsmodels.compat.python import lzip
from statsmodels.compat.pandas import Appender
import numpy as np
from scipy import stats
import pandas as pd
import patsy
from collections import defaultdict
from statsmodels.tools.decorators import cache_readonly
import statsmodels.base.model as base
import statsmodels.regression.linear_model as lm
import statsmodels.base.wrapper as wrap
from statsmodels.genmod import families
from statsmodels.genmod.generalized_linear_model import GLM, GLMResults
from statsmodels.genmod import cov_struct as cov_structs
import statsmodels.genmod.families.varfuncs as varfuncs
from statsmodels.genmod.families.links import Link
from statsmodels.tools.sm_exceptions import (ConvergenceWarning,
import warnings
from statsmodels.graphics._regressionplots_doc import (
from statsmodels.discrete.discrete_margins import (
def _score_test_submodel(par, sub):
    """
    Return transformation matrices for design matrices.

    Parameters
    ----------
    par : instance
        The parent model
    sub : instance
        The sub-model

    Returns
    -------
    qm : array_like
        Matrix mapping the design matrix of the parent to the design matrix
        for the sub-model.
    qc : array_like
        Matrix mapping the design matrix of the parent to the orthogonal
        complement of the columnspace of the submodel in the columnspace
        of the parent.

    Notes
    -----
    Returns None, None if the provided submodel is not actually a submodel.
    """
    x1 = par.exog
    x2 = sub.exog
    u, s, vt = np.linalg.svd(x1, 0)
    v = vt.T
    a, _ = np.linalg.qr(x2)
    a = u - np.dot(a, np.dot(a.T, u))
    x2c, sb, _ = np.linalg.svd(a, 0)
    x2c = x2c[:, sb > 1e-12]
    ii = np.flatnonzero(np.abs(s) > 1e-12)
    qm = np.dot(v[:, ii], np.dot(u[:, ii].T, x2) / s[ii, None])
    e = np.max(np.abs(x2 - np.dot(x1, qm)))
    if e > 1e-08:
        return (None, None)
    qc = np.dot(v[:, ii], np.dot(u[:, ii].T, x2c) / s[ii, None])
    return (qm, qc)
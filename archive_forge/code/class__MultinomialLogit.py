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
class _MultinomialLogit(Link):
    """
    The multinomial logit transform, only for use with GEE.

    Notes
    -----
    The data are assumed coded as binary indicators, where each
    observed multinomial value y is coded as I(y == S[0]), ..., I(y ==
    S[-1]), where S is the set of possible response labels, excluding
    the largest one.  Thererefore functions in this class should only
    be called using vector argument whose length is a multiple of |S|
    = ncut, which is an argument to be provided when initializing the
    class.

    call and derivative use a private method _clean to trim p by 1e-10
    so that p is in (0, 1)
    """

    def __init__(self, ncut):
        self.ncut = ncut

    def inverse(self, lpr):
        """
        Inverse of the multinomial logit transform, which gives the
        expected values of the data as a function of the linear
        predictors.

        Parameters
        ----------
        lpr : array_like (length must be divisible by `ncut`)
            The linear predictors

        Returns
        -------
        prob : ndarray
            Probabilities, or expected values
        """
        expval = np.exp(lpr)
        denom = 1 + np.reshape(expval, (len(expval) // self.ncut, self.ncut)).sum(1)
        denom = np.kron(denom, np.ones(self.ncut, dtype=np.float64))
        prob = expval / denom
        return prob
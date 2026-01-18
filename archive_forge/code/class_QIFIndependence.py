import numpy as np
from collections import defaultdict
import statsmodels.base.model as base
from statsmodels.genmod import families
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import links
from statsmodels.genmod.families import varfuncs
import statsmodels.regression.linear_model as lm
import statsmodels.base.wrapper as wrap
from statsmodels.tools.decorators import cache_readonly
class QIFIndependence(QIFCovariance):
    """
    Independent working covariance for QIF regression.  This covariance
    model gives identical results to GEE with the independence working
    covariance.  When using QIFIndependence as the working covariance,
    the QIF value will be zero, and cannot be used for chi^2 testing, or
    for model selection using AIC, BIC, etc.
    """

    def __init__(self):
        self.num_terms = 1

    def mat(self, dim, term):
        if term == 0:
            return np.eye(dim)
        else:
            return None
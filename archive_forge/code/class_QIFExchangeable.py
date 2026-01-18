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
class QIFExchangeable(QIFCovariance):
    """
    Exchangeable working covariance for QIF regression.
    """

    def __init__(self):
        self.num_terms = 2

    def mat(self, dim, term):
        if term == 0:
            return np.eye(dim)
        elif term == 1:
            return np.ones((dim, dim))
        else:
            return None
from statsmodels.compat import lrange
import os
import numpy as np
import pytest
from numpy.testing import (assert_almost_equal, assert_equal, assert_allclose,
import statsmodels.genmod.generalized_estimating_equations as gee
import statsmodels.tools as tools
import statsmodels.regression.linear_model as lm
from statsmodels.genmod import families
from statsmodels.genmod import cov_struct
import statsmodels.discrete.discrete_model as discrete
import pandas as pd
from scipy.stats.distributions import norm
import warnings
class TestGEEPoissonFormulaCovType(CheckConsistency):

    @classmethod
    def setup_class(cls):
        endog, exog, group_n = load_data('gee_poisson_1.csv')
        family = families.Poisson()
        vi = cov_struct.Independence()
        D = np.concatenate((endog[:, None], group_n[:, None], exog[:, 1:]), axis=1)
        D = pd.DataFrame(D)
        D.columns = ['Y', 'Id'] + ['X%d' % (k + 1) for k in range(exog.shape[1] - 1)]
        cls.mod = gee.GEE.from_formula('Y ~ X1 + X2 + X3 + X4 + X5', 'Id', D, family=family, cov_struct=vi)
        cls.start_params = np.array([-0.03644504, -0.05432094, 0.01566427, 0.57628591, -0.0046566, -0.47709315])
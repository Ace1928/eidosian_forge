from io import StringIO
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
import pandas as pd
import patsy
import pytest
from statsmodels import datasets
from statsmodels.base._constraints import fit_constrained
from statsmodels.discrete.discrete_model import Poisson, Logit
from statsmodels.genmod import families
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.tools.tools import add_constant
from .results import (
class TestPoissonConstrained1c(CheckPoissonConstrainedMixin):

    @classmethod
    def setup_class(cls):
        cls.res2 = results.results_exposure_constraint
        cls.idx = [6, 2, 3, 4, 5, 0]
        formula = 'deaths ~ smokes + C(agecat)'
        mod = Poisson.from_formula(formula, data=data, offset=np.log(data['pyears'].values))
        constr = 'C(agecat)[T.4] = C(agecat)[T.5]'
        lc = patsy.DesignInfo(mod.exog_names).linear_constraint(constr)
        cls.res1 = fit_constrained(mod, lc.coefs, lc.constants, fit_kwds={'method': 'newton', 'disp': 0})
        cls.constraints = lc
        cls.res1m = mod.fit_constrained(constr, method='newton', disp=0)
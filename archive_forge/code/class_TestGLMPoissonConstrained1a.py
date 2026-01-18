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
class TestGLMPoissonConstrained1a(CheckPoissonConstrainedMixin):

    @classmethod
    def setup_class(cls):
        from statsmodels.base._constraints import fit_constrained
        cls.res2 = results.results_noexposure_constraint
        cls.idx = [7, 3, 4, 5, 6, 0, 1]
        formula = 'deaths ~ logpyears + smokes + C(agecat)'
        mod = GLM.from_formula(formula, data=data, family=families.Poisson())
        constr = 'C(agecat)[T.4] = C(agecat)[T.5]'
        lc = patsy.DesignInfo(mod.exog_names).linear_constraint(constr)
        cls.res1 = fit_constrained(mod, lc.coefs, lc.constants, fit_kwds={'atol': 1e-10})
        cls.constraints = lc
        cls.res1m = mod.fit_constrained(constr, atol=1e-10)
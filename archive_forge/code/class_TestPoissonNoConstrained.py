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
class TestPoissonNoConstrained(CheckPoissonConstrainedMixin):

    @classmethod
    def setup_class(cls):
        cls.res2 = results.results_exposure_noconstraint
        cls.idx = [6, 2, 3, 4, 5, 0]
        formula = 'deaths ~ smokes + C(agecat)'
        mod = Poisson.from_formula(formula, data=data, offset=np.log(data['pyears'].values))
        res1 = mod.fit(disp=0)._results
        cls.res1 = (res1.params, res1.cov_params())
        cls.res1m = res1
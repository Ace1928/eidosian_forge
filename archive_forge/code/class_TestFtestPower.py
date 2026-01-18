import copy
import warnings
import numpy as np
from numpy.testing import (assert_almost_equal, assert_allclose, assert_raises,
import pytest
import statsmodels.stats.power as smp
from statsmodels.stats.tests.test_weightstats import Holder
from statsmodels.tools.sm_exceptions import HypothesisTestWarning
class TestFtestPower(CheckPowerMixin):

    @classmethod
    def setup_class(cls):
        res2 = Holder()
        res2.u = 5
        res2.v = 19
        res2.f2 = 0.09
        res2.sig_level = 0.1
        res2.power = 0.235454222377575
        res2.method = 'Multiple regression power calculation'
        cls.res2 = res2
        cls.kwds = {'effect_size': np.sqrt(res2.f2), 'df_num': res2.v, 'df_denom': res2.u, 'alpha': res2.sig_level, 'power': res2.power}
        cls.kwds_extra = {}
        cls.args_names = ['effect_size', 'df_num', 'df_denom', 'alpha']
        cls.cls = smp.FTestPower
        cls.decimal = 5

    def test_kwargs(self):
        with pytest.warns(UserWarning):
            smp.FTestPower().solve_power(effect_size=0.3, alpha=0.1, power=0.9, df_denom=2, nobs=None)
        with pytest.raises(ValueError):
            smp.FTestPower().solve_power(effect_size=0.3, alpha=0.1, power=0.9, df_denom=2, junk=3)
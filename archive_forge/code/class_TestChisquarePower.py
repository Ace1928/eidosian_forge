import copy
import warnings
import numpy as np
from numpy.testing import (assert_almost_equal, assert_allclose, assert_raises,
import pytest
import statsmodels.stats.power as smp
from statsmodels.stats.tests.test_weightstats import Holder
from statsmodels.tools.sm_exceptions import HypothesisTestWarning
class TestChisquarePower(CheckPowerMixin):

    @classmethod
    def setup_class(cls):
        res2 = Holder()
        res2.w = 0.1
        res2.N = 5
        res2.df = 4
        res2.sig_level = 0.05
        res2.power = 0.05246644635810126
        res2.method = 'Chi squared power calculation'
        res2.note = 'N is the number of observations'
        cls.res2 = res2
        cls.kwds = {'effect_size': res2.w, 'nobs': res2.N, 'alpha': res2.sig_level, 'power': res2.power}
        cls.kwds_extra = {'n_bins': res2.df + 1}
        cls.cls = smp.GofChisquarePower

    def test_positional(self):
        res1 = self.cls()
        args_names = ['effect_size', 'nobs', 'alpha', 'n_bins']
        kwds = copy.copy(self.kwds)
        del kwds['power']
        kwds.update(self.kwds_extra)
        args = [kwds[arg] for arg in args_names]
        if hasattr(self, 'decimal'):
            decimal = self.decimal
        else:
            decimal = 6
        assert_almost_equal(res1.power(*args), self.res2.power, decimal=decimal)
import copy
import warnings
import numpy as np
from numpy.testing import (assert_almost_equal, assert_allclose, assert_raises,
import pytest
import statsmodels.stats.power as smp
from statsmodels.stats.tests.test_weightstats import Holder
from statsmodels.tools.sm_exceptions import HypothesisTestWarning
class TestFtestAnovaPower(CheckPowerMixin):

    @classmethod
    def setup_class(cls):
        res2 = Holder()
        res2.f = 0.25
        res2.n = 200
        res2.k = 10
        res2.alpha = 0.1592
        res2.power = 0.8408
        res2.method = 'Multiple regression power calculation'
        cls.res2 = res2
        cls.kwds = {'effect_size': res2.f, 'nobs': res2.n, 'alpha': res2.alpha, 'power': res2.power}
        cls.kwds_extra = {'k_groups': res2.k}
        cls.cls = smp.FTestAnovaPower
        cls.decimal = 4
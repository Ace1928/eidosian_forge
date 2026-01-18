import copy
import warnings
import numpy as np
from numpy.testing import (assert_almost_equal, assert_allclose, assert_raises,
import pytest
import statsmodels.stats.power as smp
from statsmodels.stats.tests.test_weightstats import Holder
from statsmodels.tools.sm_exceptions import HypothesisTestWarning
class TestTTPowerTwoS3(CheckPowerMixin):

    @classmethod
    def setup_class(cls):
        res2 = Holder()
        res2.n = 30
        res2.d = 1
        res2.sig_level = 0.05
        res2.power = 0.985459690251624
        res2.alternative = 'greater'
        res2.note = 'n is number in *each* group'
        res2.method = 'Two-sample t test power calculation'
        cls.res2 = res2
        cls.kwds = {'effect_size': res2.d, 'nobs1': res2.n, 'alpha': res2.sig_level, 'power': res2.power, 'ratio': 1}
        cls.kwds_extra = {'alternative': 'larger'}
        cls.cls = smp.TTestIndPower
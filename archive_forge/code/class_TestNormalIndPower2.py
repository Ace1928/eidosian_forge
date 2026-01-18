import copy
import warnings
import numpy as np
from numpy.testing import (assert_almost_equal, assert_allclose, assert_raises,
import pytest
import statsmodels.stats.power as smp
from statsmodels.stats.tests.test_weightstats import Holder
from statsmodels.tools.sm_exceptions import HypothesisTestWarning
class TestNormalIndPower2(CheckPowerMixin):

    @classmethod
    def setup_class(cls):
        res2 = Holder()
        res2.h = 0.01
        res2.n = 80
        res2.sig_level = 0.05
        res2.power = 0.0438089705093578
        res2.alternative = 'less'
        res2.method = 'Difference of proportion power calculation for' + ' binomial distribution (arcsine transformation)'
        res2.note = 'same sample sizes'
        cls.res2 = res2
        cls.kwds = {'effect_size': res2.h, 'nobs1': res2.n, 'alpha': res2.sig_level, 'power': res2.power, 'ratio': 1}
        cls.kwds_extra = {'alternative': 'smaller'}
        cls.cls = smp.NormalIndPower
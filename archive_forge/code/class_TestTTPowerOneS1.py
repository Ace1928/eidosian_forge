import copy
import warnings
import numpy as np
from numpy.testing import (assert_almost_equal, assert_allclose, assert_raises,
import pytest
import statsmodels.stats.power as smp
from statsmodels.stats.tests.test_weightstats import Holder
from statsmodels.tools.sm_exceptions import HypothesisTestWarning
class TestTTPowerOneS1(CheckPowerMixin):

    @classmethod
    def setup_class(cls):
        res2 = Holder()
        res2.n = 30
        res2.d = 1
        res2.sig_level = 0.05
        res2.power = 0.9995636009612725
        res2.alternative = 'two.sided'
        res2.note = 'NULL'
        res2.method = 'One-sample t test power calculation'
        cls.res2 = res2
        cls.kwds = {'effect_size': res2.d, 'nobs': res2.n, 'alpha': res2.sig_level, 'power': res2.power}
        cls.kwds_extra = {}
        cls.cls = smp.TTestPower
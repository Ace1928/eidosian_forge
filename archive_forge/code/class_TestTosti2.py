import numpy as np
from numpy.testing import assert_almost_equal, assert_equal, assert_
import pytest
import statsmodels.stats.weightstats as smws
from statsmodels.tools.testing import Holder
class TestTosti2(CheckTostMixin):

    @classmethod
    def setup_class(cls):
        cls.res2 = tost_clinic_indep
        x, y = (clinic[:15, 3], clinic[15:, 3])
        cls.res1 = Holder()
        res = smws.ttost_ind(x, y, -0.6, 0.6, usevar='unequal')
        cls.res1.pvalue = res[0]
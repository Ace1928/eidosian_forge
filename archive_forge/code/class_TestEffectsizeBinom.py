import io
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_equal, assert_allclose
from statsmodels.regression.linear_model import WLS
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.stats.meta_analysis import (
from .results import results_meta
class TestEffectsizeBinom:

    @classmethod
    def setup_class(cls):
        cls.results = results_meta.eff_prop1
        ss = '            study,nei,nci,e1i,c1i,e2i,c2i,e3i,c3i,e4i,c4i\n            1,19,22,16.0,20.0,11,12,4.0,8.0,4,3\n            2,34,35,22.0,22.0,18,12,15.0,8.0,15,6\n            3,72,68,44.0,40.0,21,15,10.0,3.0,3,0\n            4,22,20,19.0,12.0,14,5,5.0,4.0,2,3\n            5,70,32,62.0,27.0,42,13,26.0,6.0,15,5\n            6,183,94,130.0,65.0,80,33,47.0,14.0,30,11\n            7,26,50,24.0,30.0,13,18,5.0,10.0,3,9\n            8,61,55,51.0,44.0,37,30,19.0,19.0,11,15\n            9,36,25,30.0,17.0,23,12,13.0,4.0,10,4\n            10,45,35,43.0,35.0,19,14,8.0,4.0,6,0\n            11,246,208,169.0,139.0,106,76,67.0,42.0,51,35\n            12,386,141,279.0,97.0,170,46,97.0,21.0,73,8\n            13,59,32,56.0,30.0,34,17,21.0,9.0,20,7\n            14,45,15,42.0,10.0,18,3,9.0,1.0,9,1\n            15,14,18,14.0,18.0,13,14,12.0,13.0,9,12\n            16,26,19,21.0,15.0,12,10,6.0,4.0,5,1\n            17,74,75,,,42,40,,,23,30'
        df3 = pd.read_csv(io.StringIO(ss))
        df_12y = df3[['e2i', 'nei', 'c2i', 'nci']]
        cls.count1, cls.nobs1, cls.count2, cls.nobs2 = df_12y.values.T

    def test_effectsize(self):
        res2 = self.results
        dta = (self.count1, self.nobs1, self.count2, self.nobs2)
        eff, var_eff = effectsize_2proportions(*dta)
        assert_allclose(eff, res2.y_rd, rtol=1e-13)
        assert_allclose(var_eff, res2.v_rd, rtol=1e-13)
        eff, var_eff = effectsize_2proportions(*dta, statistic='rr')
        assert_allclose(eff, res2.y_rr, rtol=1e-13)
        assert_allclose(var_eff, res2.v_rr, rtol=1e-13)
        eff, var_eff = effectsize_2proportions(*dta, statistic='or')
        assert_allclose(eff, res2.y_or, rtol=1e-13)
        assert_allclose(var_eff, res2.v_or, rtol=1e-13)
        eff, var_eff = effectsize_2proportions(*dta, statistic='as')
        assert_allclose(eff, res2.y_as, rtol=1e-13)
        assert_allclose(var_eff, res2.v_as, rtol=1e-13)
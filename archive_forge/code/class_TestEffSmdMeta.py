import io
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_equal, assert_allclose
from statsmodels.regression.linear_model import WLS
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.stats.meta_analysis import (
from .results import results_meta
class TestEffSmdMeta:

    @classmethod
    def setup_class(cls):
        data = [['Carroll', 94, 22, 60, 92, 20, 60], ['Grant', 98, 21, 65, 92, 22, 65], ['Peck', 98, 28, 40, 88, 26, 40], ['Donat', 94, 19, 200, 82, 17, 200], ['Stewart', 98, 21, 50, 88, 22, 45], ['Young', 96, 21, 85, 92, 22, 85]]
        colnames = ['study', 'mean_t', 'sd_t', 'n_t', 'mean_c', 'sd_c', 'n_c']
        dframe = pd.DataFrame(data, columns=colnames)
        cls.dta = np.asarray(dframe[['mean_t', 'sd_t', 'n_t', 'mean_c', 'sd_c', 'n_c']]).T
        cls.row_names = dframe['study']

    def test_smd(self):
        yi = np.array([0.09452415852032972, 0.2773558662655102, 0.36654442951592, 0.664384968326914, 0.4618062812876984, 0.18516443739910043])
        vi_asy = np.array([0.0333705617355999, 0.03106510106366112, 0.0508397176175572, 0.01055175923267344, 0.04334466980873156, 0.02363025255552155])
        vi_ub = np.array([0.03337176211751222, 0.03107388569950075, 0.05088098670518214, 0.01055698026322296, 0.04339077140867459, 0.02363252645927709])
        eff, var_eff = effectsize_smd(*self.dta)
        assert_allclose(eff, yi, rtol=1e-05)
        assert_allclose(var_eff, vi_ub, rtol=0.0001)
        assert_allclose(var_eff, vi_asy, rtol=0.002)
        yi = np.array([0.09452415852032972, 0.27735586626551023, 0.36654442951592, 0.664384968326914, 0.4612288301670527, 0.18516443739910043])
        vi_ub = np.array([0.03350541862210323, 0.03118164624093491, 0.05114625874744853, 0.0105716021428412, 0.04368303906568672, 0.02369839436451885])
        yi_m = np.array([0.0945243733606383, 0.27735640148036095, 0.3665463484579782, 0.6643850998911356, 0.46180797677414176, 0.18516464424648887])
        vi_m = np.array([0.03337182573880991, 0.03107434965484927, 0.05088322525353587, 0.01055724834741877, 0.04339324466573324, 0.0236326453714713])
        assert_allclose(eff, yi_m, rtol=1e-13)
        assert_allclose(var_eff, vi_m, rtol=1e-13)
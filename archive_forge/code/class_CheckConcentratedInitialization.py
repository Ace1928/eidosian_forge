import numpy as np
import pandas as pd
import os
import pytest
from numpy.testing import assert_, assert_equal, assert_allclose
from statsmodels.tsa.statespace.exponential_smoothing import (
class CheckConcentratedInitialization:

    @classmethod
    def setup_class(cls, mod, start_params=None, atol=0, rtol=1e-07):
        cls.start_params = start_params
        cls.atol = atol
        cls.rtol = rtol
        cls.mod = mod
        cls.conc_mod = mod.clone(mod.data.orig_endog, initialization_method='concentrated')
        cls.params = pd.Series([0.5, 0.2, 0.2, 0.95], index=['smoothing_level', 'smoothing_trend', 'smoothing_seasonal', 'damping_trend'])
        drop = []
        if not cls.mod.trend:
            drop += ['smoothing_trend', 'damping_trend']
        elif not cls.mod.damped_trend:
            drop += ['damping_trend']
        if not cls.mod.seasonal:
            drop += ['smoothing_seasonal']
        cls.params.drop(drop, inplace=True)

    def test_given_params(self):
        res = self.mod.fit_constrained(self.params.to_dict(), disp=0)
        conc_res = self.conc_mod.filter(self.params.values)
        assert_allclose(conc_res.llf, res.llf, atol=self.atol, rtol=self.rtol)
        assert_allclose(conc_res.initial_state, res.initial_state, atol=self.atol, rtol=self.rtol)

    def test_estimated_params(self):
        res = self.mod.fit(self.start_params, disp=0, maxiter=100)
        np.set_printoptions(suppress=True)
        conc_res = self.conc_mod.fit(self.start_params[:len(self.params)], disp=0)
        assert_allclose(conc_res.llf, res.llf, atol=self.atol, rtol=self.rtol)
        assert_allclose(conc_res.initial_state, res.initial_state, atol=self.atol, rtol=self.rtol)
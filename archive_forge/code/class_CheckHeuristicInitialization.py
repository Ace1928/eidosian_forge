import numpy as np
import pandas as pd
import os
import pytest
from numpy.testing import assert_, assert_equal, assert_allclose
from statsmodels.tsa.statespace.exponential_smoothing import (
class CheckHeuristicInitialization:

    @classmethod
    def setup_class(cls, mod):
        cls.mod = mod
        cls.res = cls.mod.filter(cls.mod.start_params)
        init_heuristic = np.r_[cls.mod._initial_level]
        if cls.mod.trend:
            init_heuristic = np.r_[init_heuristic, cls.mod._initial_trend]
        if cls.mod.seasonal:
            init_heuristic = np.r_[init_heuristic, cls.mod._initial_seasonal]
        cls.init_heuristic = init_heuristic
        endog = cls.mod.data.orig_endog
        initial_seasonal = cls.mod._initial_seasonal
        cls.known_mod = cls.mod.clone(endog, initialization_method='known', initial_level=cls.mod._initial_level, initial_trend=cls.mod._initial_trend, initial_seasonal=initial_seasonal)
        cls.known_res = cls.mod.filter(cls.mod.start_params)
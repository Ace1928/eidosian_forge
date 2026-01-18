import os
from collections import Counter
from itertools import combinations, product
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_array_equal
from scipy.spatial import distance
from scipy.stats import shapiro
from scipy.stats._sobol import _test_find_index
from scipy.stats import qmc
from scipy.stats._qmc import (
class TestMultinomialQMC:

    def test_validations(self):
        p = np.array([0.12, 0.26, -0.05, 0.35, 0.22])
        with pytest.raises(ValueError, match='Elements of pvals must be non-negative.'):
            qmc.MultinomialQMC(p, n_trials=10)
        p = np.array([0.12, 0.26, 0.1, 0.35, 0.22])
        message = 'Elements of pvals must sum to 1.'
        with pytest.raises(ValueError, match=message):
            qmc.MultinomialQMC(p, n_trials=10)
        p = np.array([0.12, 0.26, 0.05, 0.35, 0.22])
        message = 'Dimension of `engine` must be 1.'
        with pytest.raises(ValueError, match=message):
            qmc.MultinomialQMC(p, n_trials=10, engine=qmc.Sobol(d=2))
        message = '`engine` must be an instance of...'
        with pytest.raises(ValueError, match=message):
            qmc.MultinomialQMC(p, n_trials=10, engine=np.random.default_rng())

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_MultinomialBasicDraw(self):
        seed = np.random.default_rng(6955663962957011631562466584467607969)
        p = np.array([0.12, 0.26, 0.05, 0.35, 0.22])
        n_trials = 100
        expected = np.atleast_2d(n_trials * p).astype(int)
        engine = qmc.MultinomialQMC(p, n_trials=n_trials, seed=seed)
        assert_allclose(engine.random(1), expected, atol=1)

    def test_MultinomialDistribution(self):
        seed = np.random.default_rng(77797854505813727292048130876699859000)
        p = np.array([0.12, 0.26, 0.05, 0.35, 0.22])
        engine = qmc.MultinomialQMC(p, n_trials=8192, seed=seed)
        draws = engine.random(1)
        assert_allclose(draws / np.sum(draws), np.atleast_2d(p), atol=0.0001)

    def test_FindIndex(self):
        p_cumulative = np.array([0.1, 0.4, 0.45, 0.6, 0.75, 0.9, 0.99, 1.0])
        size = len(p_cumulative)
        assert_equal(_test_find_index(p_cumulative, size, 0.0), 0)
        assert_equal(_test_find_index(p_cumulative, size, 0.4), 2)
        assert_equal(_test_find_index(p_cumulative, size, 0.44999), 2)
        assert_equal(_test_find_index(p_cumulative, size, 0.45001), 3)
        assert_equal(_test_find_index(p_cumulative, size, 1.0), size - 1)

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_other_engine(self):
        seed = np.random.default_rng(283753519042773243071753037669078065412)
        p = np.array([0.12, 0.26, 0.05, 0.35, 0.22])
        n_trials = 100
        expected = np.atleast_2d(n_trials * p).astype(int)
        base_engine = qmc.Sobol(1, scramble=True, seed=seed)
        engine = qmc.MultinomialQMC(p, n_trials=n_trials, engine=base_engine, seed=seed)
        assert_allclose(engine.random(1), expected, atol=1)
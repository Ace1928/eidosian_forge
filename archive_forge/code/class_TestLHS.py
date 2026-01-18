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
class TestLHS(QMCEngineTests):
    qmce = qmc.LatinHypercube
    can_scramble = True

    def test_continuing(self, *args):
        pytest.skip('Not applicable: not a sequence.')

    def test_fast_forward(self, *args):
        pytest.skip('Not applicable: not a sequence.')

    def test_sample(self, *args):
        pytest.skip('Not applicable: the value of reference sample is implementation dependent.')

    @pytest.mark.parametrize('strength', [1, 2])
    @pytest.mark.parametrize('scramble', [False, True])
    @pytest.mark.parametrize('optimization', [None, 'random-CD'])
    def test_sample_stratified(self, optimization, scramble, strength):
        seed = np.random.default_rng(37511836202578819870665127532742111260)
        p = 5
        n = p ** 2
        d = 6
        engine = qmc.LatinHypercube(d=d, scramble=scramble, strength=strength, optimization=optimization, seed=seed)
        sample = engine.random(n=n)
        assert sample.shape == (n, d)
        assert engine.num_generated == n
        expected1d = (np.arange(n) + 0.5) / n
        expected = np.broadcast_to(expected1d, (d, n)).T
        assert np.any(sample != expected)
        sorted_sample = np.sort(sample, axis=0)
        tol = 0.5 / n if scramble else 0
        assert_allclose(sorted_sample, expected, atol=tol)
        assert np.any(sample - expected > tol)
        if strength == 2 and optimization is None:
            unique_elements = np.arange(p)
            desired = set(product(unique_elements, unique_elements))
            for i, j in combinations(range(engine.d), 2):
                samples_2d = sample[:, [i, j]]
                res = (samples_2d * p).astype(int)
                res_set = {tuple(row) for row in res}
                assert_equal(res_set, desired)

    def test_optimizer_1d(self):
        engine = self.engine(d=1, scramble=False)
        sample_ref = engine.random(n=64)
        optimal_ = self.engine(d=1, scramble=False, optimization='random-CD')
        sample_ = optimal_.random(n=64)
        assert_array_equal(sample_ref, sample_)

    def test_raises(self):
        message = 'not a valid strength'
        with pytest.raises(ValueError, match=message):
            qmc.LatinHypercube(1, strength=3)
        message = 'n is not the square of a prime number'
        with pytest.raises(ValueError, match=message):
            engine = qmc.LatinHypercube(d=2, strength=2)
            engine.random(16)
        message = 'n is not the square of a prime number'
        with pytest.raises(ValueError, match=message):
            engine = qmc.LatinHypercube(d=2, strength=2)
            engine.random(5)
        message = 'n is too small for d'
        with pytest.raises(ValueError, match=message):
            engine = qmc.LatinHypercube(d=5, strength=2)
            engine.random(9)
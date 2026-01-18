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
class QMCEngineTests:
    """Generic tests for QMC engines."""
    qmce = NotImplemented
    can_scramble = NotImplemented
    unscramble_nd = NotImplemented
    scramble_nd = NotImplemented
    scramble = [True, False]
    ids = ['Scrambled', 'Unscrambled']

    def engine(self, scramble: bool, seed=170382760648021597650530316304495310428, **kwargs) -> QMCEngine:
        if self.can_scramble:
            return self.qmce(scramble=scramble, seed=seed, **kwargs)
        elif scramble:
            pytest.skip()
        else:
            return self.qmce(seed=seed, **kwargs)

    def reference(self, scramble: bool) -> np.ndarray:
        return self.scramble_nd if scramble else self.unscramble_nd

    @pytest.mark.parametrize('scramble', scramble, ids=ids)
    def test_0dim(self, scramble):
        engine = self.engine(d=0, scramble=scramble)
        sample = engine.random(4)
        assert_array_equal(np.empty((4, 0)), sample)

    @pytest.mark.parametrize('scramble', scramble, ids=ids)
    def test_0sample(self, scramble):
        engine = self.engine(d=2, scramble=scramble)
        sample = engine.random(0)
        assert_array_equal(np.empty((0, 2)), sample)

    @pytest.mark.parametrize('scramble', scramble, ids=ids)
    def test_1sample(self, scramble):
        engine = self.engine(d=2, scramble=scramble)
        sample = engine.random(1)
        assert (1, 2) == sample.shape

    @pytest.mark.parametrize('scramble', scramble, ids=ids)
    def test_bounds(self, scramble):
        engine = self.engine(d=100, scramble=scramble)
        sample = engine.random(512)
        assert np.all(sample >= 0)
        assert np.all(sample <= 1)

    @pytest.mark.parametrize('scramble', scramble, ids=ids)
    def test_sample(self, scramble):
        ref_sample = self.reference(scramble=scramble)
        engine = self.engine(d=2, scramble=scramble)
        sample = engine.random(n=len(ref_sample))
        assert_allclose(sample, ref_sample, atol=0.1)
        assert engine.num_generated == len(ref_sample)

    @pytest.mark.parametrize('scramble', scramble, ids=ids)
    def test_continuing(self, scramble):
        engine = self.engine(d=2, scramble=scramble)
        ref_sample = engine.random(n=8)
        engine = self.engine(d=2, scramble=scramble)
        n_half = len(ref_sample) // 2
        _ = engine.random(n=n_half)
        sample = engine.random(n=n_half)
        assert_allclose(sample, ref_sample[n_half:], atol=0.1)

    @pytest.mark.parametrize('scramble', scramble, ids=ids)
    @pytest.mark.parametrize('seed', (170382760648021597650530316304495310428, np.random.default_rng(170382760648021597650530316304495310428), None))
    def test_reset(self, scramble, seed):
        engine = self.engine(d=2, scramble=scramble, seed=seed)
        ref_sample = engine.random(n=8)
        engine.reset()
        assert engine.num_generated == 0
        sample = engine.random(n=8)
        assert_allclose(sample, ref_sample)

    @pytest.mark.parametrize('scramble', scramble, ids=ids)
    def test_fast_forward(self, scramble):
        engine = self.engine(d=2, scramble=scramble)
        ref_sample = engine.random(n=8)
        engine = self.engine(d=2, scramble=scramble)
        engine.fast_forward(4)
        sample = engine.random(n=4)
        assert_allclose(sample, ref_sample[4:], atol=0.1)
        engine.reset()
        even_draws = []
        for i in range(8):
            if i % 2 == 0:
                even_draws.append(engine.random())
            else:
                engine.fast_forward(1)
        assert_allclose(ref_sample[[i for i in range(8) if i % 2 == 0]], np.concatenate(even_draws), atol=1e-05)

    @pytest.mark.parametrize('scramble', [True])
    def test_distribution(self, scramble):
        d = 50
        engine = self.engine(d=d, scramble=scramble)
        sample = engine.random(1024)
        assert_allclose(np.mean(sample, axis=0), np.repeat(0.5, d), atol=0.01)
        assert_allclose(np.percentile(sample, 25, axis=0), np.repeat(0.25, d), atol=0.01)
        assert_allclose(np.percentile(sample, 75, axis=0), np.repeat(0.75, d), atol=0.01)

    def test_raises_optimizer(self):
        message = "'toto' is not a valid optimization method"
        with pytest.raises(ValueError, match=message):
            self.engine(d=1, scramble=False, optimization='toto')

    @pytest.mark.parametrize('optimization,metric', [('random-CD', qmc.discrepancy), ('lloyd', lambda sample: -_l1_norm(sample))])
    def test_optimizers(self, optimization, metric):
        engine = self.engine(d=2, scramble=False)
        sample_ref = engine.random(n=64)
        metric_ref = metric(sample_ref)
        optimal_ = self.engine(d=2, scramble=False, optimization=optimization)
        sample_ = optimal_.random(n=64)
        metric_ = metric(sample_)
        assert metric_ < metric_ref

    def test_consume_prng_state(self):
        rng = np.random.default_rng(216148415951487386220755762152260027040)
        sample = []
        for i in range(3):
            engine = self.engine(d=2, scramble=True, seed=rng)
            sample.append(engine.random(4))
        with pytest.raises(AssertionError, match='Arrays are not equal'):
            assert_equal(sample[0], sample[1])
        with pytest.raises(AssertionError, match='Arrays are not equal'):
            assert_equal(sample[0], sample[2])
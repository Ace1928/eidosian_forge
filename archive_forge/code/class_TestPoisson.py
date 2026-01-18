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
class TestPoisson(QMCEngineTests):
    qmce = qmc.PoissonDisk
    can_scramble = False

    def test_bounds(self, *args):
        pytest.skip('Too costly in memory.')

    def test_fast_forward(self, *args):
        pytest.skip('Not applicable: recursive process.')

    def test_sample(self, *args):
        pytest.skip('Not applicable: the value of reference sample is implementation dependent.')

    def test_continuing(self, *args):
        radius = 0.05
        ns = 6
        engine = self.engine(d=2, radius=radius, scramble=False)
        sample_init = engine.random(n=ns)
        assert len(sample_init) <= ns
        assert l2_norm(sample_init) >= radius
        sample_continued = engine.random(n=ns)
        assert len(sample_continued) <= ns
        assert l2_norm(sample_continued) >= radius
        sample = np.concatenate([sample_init, sample_continued], axis=0)
        assert len(sample) <= ns * 2
        assert l2_norm(sample) >= radius

    def test_mindist(self):
        rng = np.random.default_rng(132074951149370773672162394161442690287)
        ns = 50
        low, high = (0.08, 0.2)
        radii = (high - low) * rng.random(5) + low
        dimensions = [1, 3, 4]
        hypersphere_methods = ['volume', 'surface']
        gen = product(dimensions, radii, hypersphere_methods)
        for d, radius, hypersphere in gen:
            engine = self.qmce(d=d, radius=radius, hypersphere=hypersphere, seed=rng)
            sample = engine.random(ns)
            assert len(sample) <= ns
            assert l2_norm(sample) >= radius

    def test_fill_space(self):
        radius = 0.2
        engine = self.qmce(d=2, radius=radius)
        sample = engine.fill_space()
        assert l2_norm(sample) >= radius

    def test_raises(self):
        message = "'toto' is not a valid hypersphere sampling"
        with pytest.raises(ValueError, match=message):
            qmc.PoissonDisk(1, hypersphere='toto')
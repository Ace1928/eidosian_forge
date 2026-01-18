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
class TestHalton(QMCEngineTests):
    qmce = qmc.Halton
    can_scramble = True
    unscramble_nd = np.array([[0, 0], [1 / 2, 1 / 3], [1 / 4, 2 / 3], [3 / 4, 1 / 9], [1 / 8, 4 / 9], [5 / 8, 7 / 9], [3 / 8, 2 / 9], [7 / 8, 5 / 9]])
    scramble_nd = np.array([[0.50246036, 0.93382481], [0.00246036, 0.26715815], [0.75246036, 0.60049148], [0.25246036, 0.8227137], [0.62746036, 0.15604704], [0.12746036, 0.48938037], [0.87746036, 0.71160259], [0.37746036, 0.04493592]])

    def test_workers(self):
        ref_sample = self.reference(scramble=True)
        engine = self.engine(d=2, scramble=True)
        sample = engine.random(n=len(ref_sample), workers=8)
        assert_allclose(sample, ref_sample, atol=0.001)
        engine.reset()
        ref_sample = engine.integers(10)
        engine.reset()
        sample = engine.integers(10, workers=8)
        assert_equal(sample, ref_sample)
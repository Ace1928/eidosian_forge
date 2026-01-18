import pytest
import numpy as np
from scipy import stats
from numpy.testing import assert_allclose, assert_array_less
from statsmodels.sandbox.distributions.extras import NormExpan_gen
class TestExpandNormMom(CheckExpandNorm):

    @classmethod
    def setup_class(cls):
        cls.scale = 2
        cls.dist1 = stats.norm(1, 2)
        cls.mvsk = [1.0, 2 ** 2, 0, 0]
        cls.dist2 = NormExpan_gen(cls.mvsk, mode='mvsk')
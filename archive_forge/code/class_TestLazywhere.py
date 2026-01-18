from multiprocessing import Pool
from multiprocessing.pool import Pool as PWL
import re
import math
from fractions import Fraction
import numpy as np
from numpy.testing import assert_equal, assert_
import pytest
from pytest import raises as assert_raises
import hypothesis.extra.numpy as npst
from hypothesis import given, strategies, reproduce_failure  # noqa: F401
from scipy.conftest import array_api_compatible
from scipy._lib._array_api import xp_assert_equal
from scipy._lib._util import (_aligned_zeros, check_random_state, MapWrapper,
class TestLazywhere:
    n_arrays = strategies.integers(min_value=1, max_value=3)
    rng_seed = strategies.integers(min_value=1000000000, max_value=9999999999)
    dtype = strategies.sampled_from((np.float32, np.float64))
    p = strategies.floats(min_value=0, max_value=1)
    data = strategies.data()

    @pytest.mark.filterwarnings('ignore::RuntimeWarning')
    @array_api_compatible
    @given(n_arrays=n_arrays, rng_seed=rng_seed, dtype=dtype, p=p, data=data)
    def test_basic(self, n_arrays, rng_seed, dtype, p, data, xp):
        mbs = npst.mutually_broadcastable_shapes(num_shapes=n_arrays + 1, min_side=0)
        input_shapes, result_shape = data.draw(mbs)
        cond_shape, *shapes = input_shapes
        fillvalue = xp.asarray(data.draw(npst.arrays(dtype=dtype, shape=tuple())))
        arrays = [xp.asarray(data.draw(npst.arrays(dtype=dtype, shape=shape))) for shape in shapes]

        def f(*args):
            return sum((arg for arg in args))

        def f2(*args):
            return sum((arg for arg in args)) / 2
        rng = np.random.default_rng(rng_seed)
        cond = xp.asarray(rng.random(size=cond_shape) > p)
        res1 = _lazywhere(cond, arrays, f, fillvalue)
        res2 = _lazywhere(cond, arrays, f, f2=f2)
        if xp == np:
            cond, fillvalue, *arrays = np.atleast_1d(cond, fillvalue, *arrays)
        ref1 = xp.where(cond, f(*arrays), fillvalue)
        ref2 = xp.where(cond, f(*arrays), f2(*arrays))
        if xp == np:
            ref1 = ref1.reshape(result_shape)
            ref2 = ref2.reshape(result_shape)
            res1 = xp.asarray(res1)[()]
            res2 = xp.asarray(res2)[()]
        isinstance(res1, type(xp.asarray([])))
        xp_assert_equal(res1, ref1)
        assert_equal(res1.shape, ref1.shape)
        assert_equal(res1.dtype, ref1.dtype)
        isinstance(res2, type(xp.asarray([])))
        xp_assert_equal(res2, ref2)
        assert_equal(res2.shape, ref2.shape)
        assert_equal(res2.dtype, ref2.dtype)
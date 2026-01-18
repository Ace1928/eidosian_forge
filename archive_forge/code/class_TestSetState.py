import hashlib
import pickle
import sys
import warnings
import numpy as np
import pytest
from numpy.testing import (
from numpy.random import MT19937, PCG64
from numpy import random
class TestSetState:

    def setup_method(self):
        self.seed = 1234567890
        self.random_state = random.RandomState(self.seed)
        self.state = self.random_state.get_state()

    def test_basic(self):
        old = self.random_state.tomaxint(16)
        self.random_state.set_state(self.state)
        new = self.random_state.tomaxint(16)
        assert_(np.all(old == new))

    def test_gaussian_reset(self):
        old = self.random_state.standard_normal(size=3)
        self.random_state.set_state(self.state)
        new = self.random_state.standard_normal(size=3)
        assert_(np.all(old == new))

    def test_gaussian_reset_in_media_res(self):
        self.random_state.standard_normal()
        state = self.random_state.get_state()
        old = self.random_state.standard_normal(size=3)
        self.random_state.set_state(state)
        new = self.random_state.standard_normal(size=3)
        assert_(np.all(old == new))

    def test_backwards_compatibility(self):
        old_state = self.state[:-2]
        x1 = self.random_state.standard_normal(size=16)
        self.random_state.set_state(old_state)
        x2 = self.random_state.standard_normal(size=16)
        self.random_state.set_state(self.state)
        x3 = self.random_state.standard_normal(size=16)
        assert_(np.all(x1 == x2))
        assert_(np.all(x1 == x3))

    def test_negative_binomial(self):
        self.random_state.negative_binomial(0.5, 0.5)

    def test_get_state_warning(self):
        rs = random.RandomState(PCG64())
        with suppress_warnings() as sup:
            w = sup.record(RuntimeWarning)
            state = rs.get_state()
            assert_(len(w) == 1)
            assert isinstance(state, dict)
            assert state['bit_generator'] == 'PCG64'

    def test_invalid_legacy_state_setting(self):
        state = self.random_state.get_state()
        new_state = ('Unknown',) + state[1:]
        assert_raises(ValueError, self.random_state.set_state, new_state)
        assert_raises(TypeError, self.random_state.set_state, np.array(new_state, dtype=object))
        state = self.random_state.get_state(legacy=False)
        del state['bit_generator']
        assert_raises(ValueError, self.random_state.set_state, state)

    def test_pickle(self):
        self.random_state.seed(0)
        self.random_state.random_sample(100)
        self.random_state.standard_normal()
        pickled = self.random_state.get_state(legacy=False)
        assert_equal(pickled['has_gauss'], 1)
        rs_unpick = pickle.loads(pickle.dumps(self.random_state))
        unpickled = rs_unpick.get_state(legacy=False)
        assert_mt19937_state_equal(pickled, unpickled)

    def test_state_setting(self):
        attr_state = self.random_state.__getstate__()
        self.random_state.standard_normal()
        self.random_state.__setstate__(attr_state)
        state = self.random_state.get_state(legacy=False)
        assert_mt19937_state_equal(attr_state, state)

    def test_repr(self):
        assert repr(self.random_state).startswith('RandomState(MT19937)')
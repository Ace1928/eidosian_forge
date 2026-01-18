import numpy as np
from itertools import product
from numpy.testing import assert_equal, assert_allclose
from pytest import raises as assert_raises
import pytest
from scipy.signal import upfirdn, firwin
from scipy.signal._upfirdn import _output_len, _upfirdn_modes
from scipy.signal._upfirdn_apply import _pad_test
def _random_factors(self, p_max, q_max, h_dtype, x_dtype):
    n_rep = 3
    longest_h = 25
    random_state = np.random.RandomState(17)
    tests = []
    for _ in range(n_rep):
        p_add = q_max if p_max > q_max else 1
        q_add = p_max if q_max > p_max else 1
        p = random_state.randint(p_max) + p_add
        q = random_state.randint(q_max) + q_add
        len_h = random_state.randint(longest_h) + 1
        h = np.atleast_1d(random_state.randint(len_h))
        h = h.astype(h_dtype)
        if h_dtype == complex:
            h += 1j * random_state.randint(len_h)
        tests.append(UpFIRDnCase(p, q, h, x_dtype))
    return tests
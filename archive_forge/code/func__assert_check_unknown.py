import pickle
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn.utils._encode import _check_unknown, _encode, _get_counts, _unique
def _assert_check_unknown(values, uniques, expected_diff, expected_mask):
    diff = _check_unknown(values, uniques)
    assert_array_equal(diff, expected_diff)
    diff, valid_mask = _check_unknown(values, uniques, return_mask=True)
    assert_array_equal(diff, expected_diff)
    assert_array_equal(valid_mask, expected_mask)
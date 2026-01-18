import numbers
import operator
import numpy as np
from numpy.testing import assert_, assert_equal, assert_raises
def _assert_equal_type_and_value(result, expected, err_msg=None):
    assert_equal(type(result), type(expected), err_msg=err_msg)
    if isinstance(result, tuple):
        assert_equal(len(result), len(expected), err_msg=err_msg)
        for result_item, expected_item in zip(result, expected):
            _assert_equal_type_and_value(result_item, expected_item, err_msg)
    else:
        assert_equal(result.value, expected.value, err_msg=err_msg)
        assert_equal(getattr(result.value, 'dtype', None), getattr(expected.value, 'dtype', None), err_msg=err_msg)
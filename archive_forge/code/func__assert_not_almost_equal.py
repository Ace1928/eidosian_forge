import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def _assert_not_almost_equal(a, b, **kwargs):
    """
    Check that two objects are not approximately equal.

    Parameters
    ----------
    a : object
        The first object to compare.
    b : object
        The second object to compare.
    **kwargs
        The arguments passed to `tm.assert_almost_equal`.
    """
    try:
        tm.assert_almost_equal(a, b, **kwargs)
        msg = f"{a} and {b} were approximately equal when they shouldn't have been"
        pytest.fail(reason=msg)
    except AssertionError:
        pass
from functools import partial
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
class TestEnsureNumeric:

    def test_numeric_values(self):
        assert nanops._ensure_numeric(1) == 1
        assert nanops._ensure_numeric(1.1) == 1.1
        assert nanops._ensure_numeric(1 + 2j) == 1 + 2j

    def test_ndarray(self):
        values = np.array([1, 2, 3])
        assert np.allclose(nanops._ensure_numeric(values), values)
        o_values = values.astype(object)
        assert np.allclose(nanops._ensure_numeric(o_values), values)
        s_values = np.array(['1', '2', '3'], dtype=object)
        msg = "Could not convert \\['1' '2' '3'\\] to numeric"
        with pytest.raises(TypeError, match=msg):
            nanops._ensure_numeric(s_values)
        s_values = np.array(['foo', 'bar', 'baz'], dtype=object)
        msg = 'Could not convert .* to numeric'
        with pytest.raises(TypeError, match=msg):
            nanops._ensure_numeric(s_values)

    def test_convertable_values(self):
        with pytest.raises(TypeError, match="Could not convert string '1' to numeric"):
            nanops._ensure_numeric('1')
        with pytest.raises(TypeError, match="Could not convert string '1.1' to numeric"):
            nanops._ensure_numeric('1.1')
        with pytest.raises(TypeError, match="Could not convert string '1\\+1j' to numeric"):
            nanops._ensure_numeric('1+1j')

    def test_non_convertable_values(self):
        msg = "Could not convert string 'foo' to numeric"
        with pytest.raises(TypeError, match=msg):
            nanops._ensure_numeric('foo')
        msg = 'argument must be a string or a number'
        with pytest.raises(TypeError, match=msg):
            nanops._ensure_numeric({})
        with pytest.raises(TypeError, match=msg):
            nanops._ensure_numeric([])
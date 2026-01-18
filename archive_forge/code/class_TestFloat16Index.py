import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
class TestFloat16Index:

    def test_constructor(self):
        index_cls = Index
        dtype = np.float16
        msg = 'float16 indexes are not supported'
        with pytest.raises(NotImplementedError, match=msg):
            index_cls([1, 2, 3, 4, 5], dtype=dtype)
        with pytest.raises(NotImplementedError, match=msg):
            index_cls(np.array([1, 2, 3, 4, 5]), dtype=dtype)
        with pytest.raises(NotImplementedError, match=msg):
            index_cls([1.0, 2, 3, 4, 5], dtype=dtype)
        with pytest.raises(NotImplementedError, match=msg):
            index_cls(np.array([1.0, 2, 3, 4, 5]), dtype=dtype)
        with pytest.raises(NotImplementedError, match=msg):
            index_cls([1.0, 2, 3, 4, 5], dtype=dtype)
        with pytest.raises(NotImplementedError, match=msg):
            index_cls(np.array([1.0, 2, 3, 4, 5]), dtype=dtype)
        with pytest.raises(NotImplementedError, match=msg):
            index_cls([np.nan, np.nan], dtype=dtype)
        with pytest.raises(NotImplementedError, match=msg):
            index_cls(np.array([np.nan]), dtype=dtype)
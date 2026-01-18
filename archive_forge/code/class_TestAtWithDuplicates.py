from datetime import (
import numpy as np
import pytest
from pandas.errors import InvalidIndexError
from pandas import (
import pandas._testing as tm
class TestAtWithDuplicates:

    def test_at_with_duplicate_axes_requires_scalar_lookup(self):
        arr = np.random.default_rng(2).standard_normal(6).reshape(3, 2)
        df = DataFrame(arr, columns=['A', 'A'])
        msg = 'Invalid call for scalar access'
        with pytest.raises(ValueError, match=msg):
            df.at[[1, 2]]
        with pytest.raises(ValueError, match=msg):
            df.at[1, ['A']]
        with pytest.raises(ValueError, match=msg):
            df.at[:, 'A']
        with pytest.raises(ValueError, match=msg):
            df.at[[1, 2]] = 1
        with pytest.raises(ValueError, match=msg):
            df.at[1, ['A']] = 1
        with pytest.raises(ValueError, match=msg):
            df.at[:, 'A'] = 1
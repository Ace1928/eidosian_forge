import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
class TestKeyErrorsWithMultiIndex:

    def test_missing_keys_raises_keyerror(self):
        df = DataFrame(np.arange(12).reshape(4, 3), columns=['A', 'B', 'C'])
        df2 = df.set_index(['A', 'B'])
        with pytest.raises(KeyError, match='1'):
            df2.loc[1, 6]

    def test_missing_key_raises_keyerror2(self):
        ser = Series(-1, index=MultiIndex.from_product([[0, 1]] * 2))
        with pytest.raises(KeyError, match='\\(0, 3\\)'):
            ser.loc[0, 3]

    def test_missing_key_combination(self):
        mi = MultiIndex.from_arrays([np.array(['a', 'a', 'b', 'b']), np.array(['1', '2', '2', '3']), np.array(['c', 'd', 'c', 'd'])], names=['one', 'two', 'three'])
        df = DataFrame(np.random.default_rng(2).random((4, 3)), index=mi)
        msg = "\\('b', '1', slice\\(None, None, None\\)\\)"
        with pytest.raises(KeyError, match=msg):
            df.loc[('b', '1', slice(None)), :]
        with pytest.raises(KeyError, match=msg):
            df.index.get_locs(('b', '1', slice(None)))
        with pytest.raises(KeyError, match="\\('b', '1'\\)"):
            df.loc[('b', '1'), :]
import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import SparseArray
class TestGetitemDeprecatedIndexers:

    @pytest.mark.parametrize('key', [{'a', 'b'}, {'a': 'a'}])
    def test_getitem_dict_and_set_deprecated(self, key):
        df = DataFrame([[1, 2], [3, 4]], columns=MultiIndex.from_tuples([('a', 1), ('b', 2)]))
        with pytest.raises(TypeError, match='as an indexer is not supported'):
            df[key]
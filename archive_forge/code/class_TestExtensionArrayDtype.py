import numpy as np
import pytest
from pandas.core.dtypes import dtypes
from pandas.core.dtypes.common import is_extension_array_dtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import ExtensionArray
class TestExtensionArrayDtype:

    @pytest.mark.parametrize('values', [pd.Categorical([]), pd.Categorical([]).dtype, pd.Series(pd.Categorical([])), DummyDtype(), DummyArray(np.array([1, 2]))])
    def test_is_extension_array_dtype(self, values):
        assert is_extension_array_dtype(values)

    @pytest.mark.parametrize('values', [np.array([]), pd.Series(np.array([]))])
    def test_is_not_extension_array_dtype(self, values):
        assert not is_extension_array_dtype(values)
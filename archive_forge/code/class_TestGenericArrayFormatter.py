from datetime import datetime
from io import StringIO
from pathlib import Path
import re
from shutil import get_terminal_size
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
from pandas import (
from pandas.io.formats import printing
import pandas.io.formats.format as fmt
class TestGenericArrayFormatter:

    def test_1d_array(self):
        obj = fmt._GenericArrayFormatter(np.array([True, False]))
        res = obj.get_result()
        assert len(res) == 2
        assert res[0] == '  True'
        assert res[1] == ' False'

    def test_2d_array(self):
        obj = fmt._GenericArrayFormatter(np.array([[True, False], [False, True]]))
        res = obj.get_result()
        assert len(res) == 2
        assert res[0] == ' [True, False]'
        assert res[1] == ' [False, True]'

    def test_3d_array(self):
        obj = fmt._GenericArrayFormatter(np.array([[[True, True], [False, False]], [[False, True], [True, False]]]))
        res = obj.get_result()
        assert len(res) == 2
        assert res[0] == ' [[True, True], [False, False]]'
        assert res[1] == ' [[False, True], [True, False]]'

    def test_2d_extension_type(self):

        class DtypeStub(pd.api.extensions.ExtensionDtype):

            @property
            def type(self):
                return np.ndarray

            @property
            def name(self):
                return 'DtypeStub'

        class ExtTypeStub(pd.api.extensions.ExtensionArray):

            def __len__(self) -> int:
                return 2

            def __getitem__(self, ix):
                return [ix == 1, ix == 0]

            @property
            def dtype(self):
                return DtypeStub()
        series = Series(ExtTypeStub(), copy=False)
        res = repr(series)
        expected = '\n'.join(['0    [False True]', '1    [True False]', 'dtype: DtypeStub'])
        assert res == expected
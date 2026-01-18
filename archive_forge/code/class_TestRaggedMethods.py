from __future__ import annotations
import pytest
import numpy as np
import pandas as pd
import pandas.tests.extension.base as eb
from packaging.version import Version
from datashader.datatypes import RaggedDtype, RaggedArray
from pandas.tests.extension.conftest import *  # noqa (fixture import)
class TestRaggedMethods(eb.BaseMethodsTests):

    @pytest.mark.skip(reason='value_counts not supported')
    def test_value_counts(self):
        pass

    @pytest.mark.skip(reason='value_counts not supported')
    def test_value_counts_with_normalize(self):
        pass

    @pytest.mark.skip(reason='shift not supported')
    def test_shift_0_periods(self):
        pass

    @pytest.mark.parametrize('box', [pd.Series, lambda x: x])
    @pytest.mark.parametrize('method', [lambda x: x.unique(), pd.unique])
    def test_unique(self, data, box, method):
        duplicated = box(data._from_sequence([data[0], data[0]]))
        result = method(duplicated)
        assert len(result) == 1
        assert isinstance(result, type(data))
        np.testing.assert_array_equal(result[0], duplicated[0])

    @pytest.mark.skip(reason='pandas cannot fill with ndarray')
    def test_fillna_copy_frame(self):
        pass

    @pytest.mark.skip(reason='pandas cannot fill with ndarray')
    def test_fillna_copy_series(self):
        pass

    @pytest.mark.skip(reason='ragged does not support <= on elements')
    def test_combine_le(self):
        pass

    @pytest.mark.skip(reason='ragged does not support + on elements')
    def test_combine_add(self):
        pass

    @pytest.mark.skip(reason='combine_first not supported')
    def test_combine_first(self):
        pass

    @pytest.mark.skip(reason='Searchsorted seems not implemented for custom extension arrays')
    def test_searchsorted(self):
        pass

    @pytest.mark.skip(reason='ragged cannot be used as categorical')
    def test_sort_values_frame(self):
        pass

    @pytest.mark.skip(reason='__setitem__ not supported')
    def test_where_series(self):
        pass

    @pytest.mark.xfail(reason='not currently supported')
    def test_duplicated(self, data):
        super().test_duplicated(data)
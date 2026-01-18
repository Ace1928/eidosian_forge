import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.indexes.datetimelike import DatetimeLike
class TestDatetimeIndex(DatetimeLike):
    _index_cls = DatetimeIndex

    @pytest.fixture
    def simple_index(self) -> DatetimeIndex:
        return date_range('20130101', periods=5)

    @pytest.fixture(params=[tm.makeDateIndex(10), date_range('20130110', periods=10, freq='-1D')], ids=['index_inc', 'index_dec'])
    def index(self, request):
        return request.param

    def test_format(self, simple_index):
        idx = simple_index
        expected = [f'{x:%Y-%m-%d}' for x in idx]
        assert idx.format() == expected

    def test_shift(self):
        pass

    def test_intersection(self):
        pass

    def test_union(self):
        pass
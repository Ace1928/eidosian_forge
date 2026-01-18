import pytest
from pandas import Timedelta
class TestReprBase:

    def test_none(self):
        delta_1d = Timedelta(1, unit='D')
        delta_0d = Timedelta(0, unit='D')
        delta_1s = Timedelta(1, unit='s')
        delta_500ms = Timedelta(500, unit='ms')
        drepr = lambda x: x._repr_base()
        assert drepr(delta_1d) == '1 days'
        assert drepr(-delta_1d) == '-1 days'
        assert drepr(delta_0d) == '0 days'
        assert drepr(delta_1s) == '0 days 00:00:01'
        assert drepr(delta_500ms) == '0 days 00:00:00.500000'
        assert drepr(delta_1d + delta_1s) == '1 days 00:00:01'
        assert drepr(-delta_1d + delta_1s) == '-1 days +00:00:01'
        assert drepr(delta_1d + delta_500ms) == '1 days 00:00:00.500000'
        assert drepr(-delta_1d + delta_500ms) == '-1 days +00:00:00.500000'

    def test_sub_day(self):
        delta_1d = Timedelta(1, unit='D')
        delta_0d = Timedelta(0, unit='D')
        delta_1s = Timedelta(1, unit='s')
        delta_500ms = Timedelta(500, unit='ms')
        drepr = lambda x: x._repr_base(format='sub_day')
        assert drepr(delta_1d) == '1 days'
        assert drepr(-delta_1d) == '-1 days'
        assert drepr(delta_0d) == '00:00:00'
        assert drepr(delta_1s) == '00:00:01'
        assert drepr(delta_500ms) == '00:00:00.500000'
        assert drepr(delta_1d + delta_1s) == '1 days 00:00:01'
        assert drepr(-delta_1d + delta_1s) == '-1 days +00:00:01'
        assert drepr(delta_1d + delta_500ms) == '1 days 00:00:00.500000'
        assert drepr(-delta_1d + delta_500ms) == '-1 days +00:00:00.500000'

    def test_long(self):
        delta_1d = Timedelta(1, unit='D')
        delta_0d = Timedelta(0, unit='D')
        delta_1s = Timedelta(1, unit='s')
        delta_500ms = Timedelta(500, unit='ms')
        drepr = lambda x: x._repr_base(format='long')
        assert drepr(delta_1d) == '1 days 00:00:00'
        assert drepr(-delta_1d) == '-1 days +00:00:00'
        assert drepr(delta_0d) == '0 days 00:00:00'
        assert drepr(delta_1s) == '0 days 00:00:01'
        assert drepr(delta_500ms) == '0 days 00:00:00.500000'
        assert drepr(delta_1d + delta_1s) == '1 days 00:00:01'
        assert drepr(-delta_1d + delta_1s) == '-1 days +00:00:01'
        assert drepr(delta_1d + delta_500ms) == '1 days 00:00:00.500000'
        assert drepr(-delta_1d + delta_500ms) == '-1 days +00:00:00.500000'

    def test_all(self):
        delta_1d = Timedelta(1, unit='D')
        delta_0d = Timedelta(0, unit='D')
        delta_1ns = Timedelta(1, unit='ns')
        drepr = lambda x: x._repr_base(format='all')
        assert drepr(delta_1d) == '1 days 00:00:00.000000000'
        assert drepr(-delta_1d) == '-1 days +00:00:00.000000000'
        assert drepr(delta_0d) == '0 days 00:00:00.000000000'
        assert drepr(delta_1ns) == '0 days 00:00:00.000000001'
        assert drepr(-delta_1d + delta_1ns) == '-1 days +00:00:00.000000001'
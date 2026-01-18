import unittest
import aniso8601
from aniso8601.builders import (
from aniso8601.exceptions import ISOFormatError
from aniso8601.interval import (
from aniso8601.resolution import IntervalResolution
from aniso8601.tests.compat import mock
class TestIntervalParser_UtilityFunctions(unittest.TestCase):

    def test_get_interval_resolution(self):
        self.assertEqual(_get_interval_resolution(IntervalTuple(start=DateTuple(YYYY='2001', MM='02', DD='03', Www=None, D=None, DDD=None), end=DatetimeTuple(DateTuple(YYYY='2001', MM='02', DD='03', Www=None, D=None, DDD=None), TimeTuple(hh='04', mm='05', ss='06', tz=None)), duration=None)), IntervalResolution.Seconds)
        self.assertEqual(_get_interval_resolution(IntervalTuple(start=DatetimeTuple(DateTuple(YYYY='2001', MM='02', DD='03', Www=None, D=None, DDD=None), TimeTuple(hh='04', mm='05', ss='06', tz=None)), end=DateTuple(YYYY='2001', MM='02', DD='03', Www=None, D=None, DDD=None), duration=None)), IntervalResolution.Seconds)
        self.assertEqual(_get_interval_resolution(IntervalTuple(start=DateTuple(YYYY='2001', MM='02', DD='03', Www=None, D=None, DDD=None), end=None, duration=DurationTuple(PnY='1', PnM='2', PnW=None, PnD='3', TnH='4', TnM='5', TnS='6'))), IntervalResolution.Seconds)
        self.assertEqual(_get_interval_resolution(IntervalTuple(start=DatetimeTuple(DateTuple(YYYY='2001', MM='02', DD='03', Www=None, D=None, DDD=None), TimeTuple(hh='04', mm='05', ss='06', tz=None)), end=None, duration=DurationTuple(PnY='1', PnM='2', PnW=None, PnD='3', TnH=None, TnM=None, TnS=None))), IntervalResolution.Seconds)
        self.assertEqual(_get_interval_resolution(IntervalTuple(start=None, end=DateTuple(YYYY='2001', MM='02', DD='03', Www=None, D=None, DDD=None), duration=DurationTuple(PnY='1', PnM='2', PnW=None, PnD='3', TnH='4', TnM='5', TnS='6'))), IntervalResolution.Seconds)
        self.assertEqual(_get_interval_resolution(IntervalTuple(start=None, end=DatetimeTuple(DateTuple(YYYY='2001', MM='02', DD='03', Www=None, D=None, DDD=None), TimeTuple(hh='04', mm='05', ss='06', tz=None)), duration=DurationTuple(PnY='1', PnM='2', PnW=None, PnD='3', TnH=None, TnM=None, TnS=None))), IntervalResolution.Seconds)

    def test_get_interval_component_resolution(self):
        self.assertEqual(_get_interval_component_resolution(DateTuple(YYYY='2001', MM=None, DD=None, Www=None, D=None, DDD='123')), IntervalResolution.Ordinal)
        self.assertEqual(_get_interval_component_resolution(DateTuple(YYYY='2001', MM=None, DD=None, Www='12', D='3', DDD=None)), IntervalResolution.Weekday)
        self.assertEqual(_get_interval_component_resolution(DateTuple(YYYY='2001', MM=None, DD=None, Www='12', D=None, DDD=None)), IntervalResolution.Week)
        self.assertEqual(_get_interval_component_resolution(DateTuple(YYYY='2001', MM='02', DD='03', Www=None, D=None, DDD=None)), IntervalResolution.Day)
        self.assertEqual(_get_interval_component_resolution(DateTuple(YYYY='2001', MM='02', DD=None, Www=None, D=None, DDD=None)), IntervalResolution.Month)
        self.assertEqual(_get_interval_component_resolution(DateTuple(YYYY='2001', MM=None, DD=None, Www=None, D=None, DDD=None)), IntervalResolution.Year)
        self.assertEqual(_get_interval_component_resolution(DatetimeTuple(DateTuple(YYYY='2001', MM='02', DD='03', Www=None, D=None, DDD=None), TimeTuple(hh='04', mm='05', ss='06', tz=None))), IntervalResolution.Seconds)
        self.assertEqual(_get_interval_component_resolution(DatetimeTuple(DateTuple(YYYY='2001', MM='02', DD='03', Www=None, D=None, DDD=None), TimeTuple(hh='04', mm='05', ss=None, tz=None))), IntervalResolution.Minutes)
        self.assertEqual(_get_interval_component_resolution(DatetimeTuple(DateTuple(YYYY='2001', MM='02', DD='03', Www=None, D=None, DDD=None), TimeTuple(hh='04', mm=None, ss=None, tz=None))), IntervalResolution.Hours)
        self.assertEqual(_get_interval_component_resolution(DurationTuple(PnY='1', PnM='2', PnW=None, PnD='3', TnH='4', TnM='5', TnS='6')), IntervalResolution.Seconds)
        self.assertEqual(_get_interval_component_resolution(DurationTuple(PnY='1', PnM='2', PnW=None, PnD='3', TnH='4', TnM='5', TnS=None)), IntervalResolution.Minutes)
        self.assertEqual(_get_interval_component_resolution(DurationTuple(PnY='1', PnM='2', PnW=None, PnD='3', TnH='4', TnM=None, TnS=None)), IntervalResolution.Hours)
        self.assertEqual(_get_interval_component_resolution(DurationTuple(PnY='1', PnM='2', PnW=None, PnD='3', TnH=None, TnM=None, TnS=None)), IntervalResolution.Day)
        self.assertEqual(_get_interval_component_resolution(DurationTuple(PnY='1', PnM='2', PnW=None, PnD=None, TnH=None, TnM=None, TnS=None)), IntervalResolution.Month)
        self.assertEqual(_get_interval_component_resolution(DurationTuple(PnY='1', PnM=None, PnW=None, PnD=None, TnH=None, TnM=None, TnS=None)), IntervalResolution.Year)
        self.assertEqual(_get_interval_component_resolution(DurationTuple(PnY=None, PnM=None, PnW='3', PnD=None, TnH=None, TnM=None, TnS=None)), IntervalResolution.Week)
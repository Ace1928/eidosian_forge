import datetime
import math
import unittest
from itertools import product
import numpy as np
import pandas as pd
from holoviews import Dimension, Element
from holoviews.core.util import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.streams import PointerXY
class TestDatetimeUtils(unittest.TestCase):

    def test_compute_density_float(self):
        self.assertEqual(compute_density(0, 1, 10), 10)

    def test_compute_us_density_1s_datetime(self):
        start = np.datetime64(datetime.datetime.today())
        end = start + np.timedelta64(1, 's')
        self.assertEqual(compute_density(start, end, 10), 1e-05)

    def test_compute_us_density_10s_datetime(self):
        start = np.datetime64(datetime.datetime.today())
        end = start + np.timedelta64(10, 's')
        self.assertEqual(compute_density(start, end, 10), 1e-06)

    def test_compute_s_density_1s_datetime(self):
        start = np.datetime64(datetime.datetime.today())
        end = start + np.timedelta64(1, 's')
        self.assertEqual(compute_density(start, end, 10, 's'), 10)

    def test_compute_s_density_10s_datetime(self):
        start = np.datetime64(datetime.datetime.today())
        end = start + np.timedelta64(10, 's')
        self.assertEqual(compute_density(start, end, 10, 's'), 1)

    def test_datetime_to_us_int(self):
        dt = datetime.datetime(2017, 1, 1)
        self.assertEqual(dt_to_int(dt), 1483228800000000.0)

    def test_datetime64_s_to_ns_int(self):
        dt = np.datetime64(datetime.datetime(2017, 1, 1), 's')
        self.assertEqual(dt_to_int(dt, 'ns'), 1.4832288e+18)

    def test_datetime64_us_to_ns_int(self):
        dt = np.datetime64(datetime.datetime(2017, 1, 1), 'us')
        self.assertEqual(dt_to_int(dt, 'ns'), 1.4832288e+18)

    def test_datetime64_to_ns_int(self):
        dt = np.datetime64(datetime.datetime(2017, 1, 1))
        self.assertEqual(dt_to_int(dt, 'ns'), 1.4832288e+18)

    def test_datetime64_us_to_us_int(self):
        dt = np.datetime64(datetime.datetime(2017, 1, 1), 'us')
        self.assertEqual(dt_to_int(dt), 1483228800000000.0)

    def test_datetime64_s_to_us_int(self):
        dt = np.datetime64(datetime.datetime(2017, 1, 1), 's')
        self.assertEqual(dt_to_int(dt), 1483228800000000.0)

    def test_timestamp_to_us_int(self):
        dt = pd.Timestamp(datetime.datetime(2017, 1, 1))
        self.assertEqual(dt_to_int(dt), 1483228800000000.0)

    def test_datetime_to_s_int(self):
        dt = datetime.datetime(2017, 1, 1)
        self.assertEqual(dt_to_int(dt, 's'), 1483228800.0)

    def test_datetime64_to_s_int(self):
        dt = np.datetime64(datetime.datetime(2017, 1, 1))
        self.assertEqual(dt_to_int(dt, 's'), 1483228800.0)

    def test_datetime64_us_to_s_int(self):
        dt = np.datetime64(datetime.datetime(2017, 1, 1), 'us')
        self.assertEqual(dt_to_int(dt, 's'), 1483228800.0)

    def test_datetime64_s_to_s_int(self):
        dt = np.datetime64(datetime.datetime(2017, 1, 1), 's')
        self.assertEqual(dt_to_int(dt, 's'), 1483228800.0)

    def test_timestamp_to_s_int(self):
        dt = pd.Timestamp(datetime.datetime(2017, 1, 1))
        self.assertEqual(dt_to_int(dt, 's'), 1483228800.0)

    def test_date_range_1_hour(self):
        start = np.datetime64(datetime.datetime(2017, 1, 1))
        end = start + np.timedelta64(1, 'h')
        drange = date_range(start, end, 6)
        self.assertEqual(drange[0], start + np.timedelta64(5, 'm'))
        self.assertEqual(drange[-1], end - np.timedelta64(5, 'm'))

    def test_date_range_1_sec(self):
        start = np.datetime64(datetime.datetime(2017, 1, 1))
        end = start + np.timedelta64(1, 's')
        drange = date_range(start, end, 10)
        self.assertEqual(drange[0], start + np.timedelta64(50, 'ms'))
        self.assertEqual(drange[-1], end - np.timedelta64(50, 'ms'))

    def test_timezone_to_int(self):
        import pytz
        timezone = pytz.timezone('Europe/Copenhagen')
        values = [datetime.datetime(2021, 4, 8, 12, 0, 0, 0), datetime.datetime(2021, 4, 8, 12, 0, 0, 0, datetime.timezone.utc), datetime.datetime(2021, 4, 8, 12, 0, 0, 0, timezone), datetime.date(2021, 4, 8), np.datetime64(datetime.datetime(2021, 4, 8, 12, 0, 0, 0))]
        for value in values:
            x1 = dt_to_int(value)
            x2 = dt_to_int(pd.to_datetime(value))
            self.assertEqual(x1, x2)
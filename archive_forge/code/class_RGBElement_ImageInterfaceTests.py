import datetime as dt
from unittest import SkipTest
import numpy as np
from holoviews import HSV, RGB, Curve, Dataset, Dimension, Image, Table
from holoviews.core.data.interface import DataError
from holoviews.core.util import date_range
from .base import DatatypeContext, GriddedInterfaceTests, InterfaceTests
class RGBElement_ImageInterfaceTests(BaseRGBElementInterfaceTests):
    datatype = 'image'
    __test__ = True

    def test_reduce_to_single_values(self):
        try:
            super().test_reduce_to_single_values()
        except DataError:
            msg = "RGB element can't run with this command: 'pytest holoviews/tests -k test_reduce_to_single_values'but runs fine with 'pytest holoviews/tests/core/'"
            raise SkipTest(msg)
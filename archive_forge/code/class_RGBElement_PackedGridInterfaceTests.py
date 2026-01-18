import datetime as dt
from itertools import product
from unittest import SkipTest
import numpy as np
import pandas as pd
from holoviews.core.data import Dataset
from holoviews.core.data.interface import DataError
from holoviews.core.util import date_range
from holoviews.element import HSV, RGB, Curve, Image
from holoviews.util.transform import dim
from .base import (
from .test_imageinterface import (
class RGBElement_PackedGridInterfaceTests(BaseRGBElementInterfaceTests):
    datatype = 'grid'
    data_type = dict
    __test__ = True

    def init_data(self):
        self.rgb = RGB((self.xs, self.ys, self.rgb_array))
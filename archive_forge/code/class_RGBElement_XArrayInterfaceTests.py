import datetime as dt
from unittest import SkipTest
import numpy as np
import pandas as pd
from holoviews.core.data import Dataset, XArrayInterface, concat
from holoviews.core.dimension import Dimension
from holoviews.core.spaces import HoloMap
from holoviews.element import HSV, RGB, Image, ImageStack, QuadMesh
from .test_gridinterface import BaseGridInterfaceTests
from .test_imageinterface import (
class RGBElement_XArrayInterfaceTests(BaseRGBElementInterfaceTests):
    datatype = 'xarray'
    data_type = xr.Dataset
    __test__ = True

    def init_data(self):
        self.rgb = RGB((self.xs, self.ys, self.rgb_array[:, :, 0], self.rgb_array[:, :, 1], self.rgb_array[:, :, 2]))
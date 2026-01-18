import datetime as dt
from unittest import SkipTest
import numpy as np
from holoviews import HSV, RGB, Curve, Dataset, Dimension, Image, Table
from holoviews.core.data.interface import DataError
from holoviews.core.util import date_range
from .base import DatatypeContext, GriddedInterfaceTests, InterfaceTests
class ImageElement_ImageInterfaceTests(BaseImageElementInterfaceTests):
    datatype = 'image'
    data_type = np.ndarray
    __test__ = True
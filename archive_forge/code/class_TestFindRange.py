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
class TestFindRange(unittest.TestCase):
    """
    Tests for find_range function.
    """

    def setUp(self):
        self.int_vals = [1, 5, 3, 9, 7, 121, 14]
        self.float_vals = [0.38, 0.121, -0.1424, 5.12]
        self.nan_floats = [np.nan, 0.32, 1.42, -0.32]
        self.str_vals = ['Aardvark', 'Zebra', 'Platypus', 'Wallaby']

    def test_int_range(self):
        self.assertEqual(find_range(self.int_vals), (1, 121))

    def test_float_range(self):
        self.assertEqual(find_range(self.float_vals), (-0.1424, 5.12))

    def test_nan_range(self):
        self.assertEqual(find_range(self.nan_floats), (-0.32, 1.42))

    def test_str_range(self):
        self.assertEqual(find_range(self.str_vals), ('Aardvark', 'Zebra'))

    def test_soft_range(self):
        self.assertEqual(find_range(self.float_vals, soft_range=(np.nan, 100)), (-0.1424, 100))
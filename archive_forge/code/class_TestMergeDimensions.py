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
class TestMergeDimensions(unittest.TestCase):

    def test_merge_dimensions(self):
        dimensions = merge_dimensions([[Dimension('A')], [Dimension('A'), Dimension('B')]])
        self.assertEqual(dimensions, [Dimension('A'), Dimension('B')])

    def test_merge_dimensions_with_values(self):
        dimensions = merge_dimensions([[Dimension('A', values=[0, 1])], [Dimension('A', values=[1, 2]), Dimension('B')]])
        self.assertEqual(dimensions, [Dimension('A'), Dimension('B')])
        self.assertEqual(dimensions[0].values, [0, 1, 2])
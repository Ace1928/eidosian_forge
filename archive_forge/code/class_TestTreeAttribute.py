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
class TestTreeAttribute(ComparisonTestCase):

    def test_simple_lowercase_string(self):
        self.assertEqual(tree_attribute('lowercase'), False)

    def test_simple_uppercase_string(self):
        self.assertEqual(tree_attribute('UPPERCASE'), True)

    def test_underscore_string(self):
        self.assertEqual(tree_attribute('_underscore'), False)
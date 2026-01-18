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
class TestAllowablePrefix(ComparisonTestCase):
    """
    Tests of allowable and hasprefix method.
    """

    def test_allowable_false_1(self):
        self.assertEqual(sanitize_identifier.allowable('trait_names'), False)

    def test_allowable_false_2(self):
        self.assertEqual(sanitize_identifier.allowable('_repr_png_'), False)

    def test_allowable_false_3(self):
        self.assertEqual(sanitize_identifier.allowable('_ipython_display_'), False)

    def test_allowable_false_underscore(self):
        self.assertEqual(sanitize_identifier.allowable('_foo', True), False)

    def test_allowable_true(self):
        self.assertEqual(sanitize_identifier.allowable('some_string'), True)

    def test_prefix_test1(self):
        prefixed = sanitize_identifier.prefixed('_some_string')
        self.assertEqual(prefixed, True)

    def test_prefix_test2(self):
        prefixed = sanitize_identifier.prefixed('some_string')
        self.assertEqual(prefixed, False)

    def test_prefix_test3(self):
        prefixed = sanitize_identifier.prefixed('۵some_string')
        self.assertEqual(prefixed, True)
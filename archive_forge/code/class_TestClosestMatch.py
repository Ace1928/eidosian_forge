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
class TestClosestMatch(ComparisonTestCase):

    def test_complete_match_overlay(self):
        specs = [(0, ('Curve', 'Curve', 'I')), (1, ('Points', 'Points', 'I'))]
        spec = ('Curve', 'Curve', 'I')
        self.assertEqual(closest_match(spec, specs), 0)
        spec = ('Points', 'Curve', 'I')
        self.assertEqual(closest_match(spec, specs), 1)

    def test_partial_match_overlay(self):
        specs = [(0, ('Curve', 'Curve', 'I')), (1, ('Points', 'Points', 'I'))]
        spec = ('Curve', 'Curve')
        self.assertEqual(closest_match(spec, specs), 0)
        spec = ('Points', 'Points')
        self.assertEqual(closest_match(spec, specs), 1)

    def test_partial_mismatch_overlay(self):
        specs = [(0, ('Curve', 'Curve', 'I')), (1, ('Points', 'Points', 'I'))]
        spec = ('Curve', 'Foo', 'II')
        self.assertEqual(closest_match(spec, specs), 0)
        spec = ('Points', 'Bar', 'III')
        self.assertEqual(closest_match(spec, specs), 1)

    def test_no_match_overlay(self):
        specs = [(0, ('Curve', 'Curve', 'I')), (1, ('Points', 'Points', 'I'))]
        spec = ('Scatter', 'Points', 'II')
        self.assertEqual(closest_match(spec, specs), None)
        spec = ('Path', 'Curve', 'III')
        self.assertEqual(closest_match(spec, specs), None)

    def test_complete_match_ndoverlay(self):
        spec = ('Points', 'Points', '', 1)
        specs = [(0, ('Points', 'Points', '', 0)), (1, ('Points', 'Points', '', 1)), (2, ('Points', 'Points', '', 2))]
        self.assertEqual(closest_match(spec, specs), 1)
        spec = ('Points', 'Points', '', 2)
        self.assertEqual(closest_match(spec, specs), 2)

    def test_partial_match_ndoverlay(self):
        spec = ('Points', 'Points', '', 5)
        specs = [(0, ('Points', 'Points', '', 0)), (1, ('Points', 'Points', '', 1)), (2, ('Points', 'Points', '', 2))]
        self.assertEqual(closest_match(spec, specs), 2)
        spec = ('Points', 'Points', 'Bar', 5)
        self.assertEqual(closest_match(spec, specs), 0)
        spec = ('Points', 'Foo', 'Bar', 5)
        self.assertEqual(closest_match(spec, specs), 0)

    def test_no_match_ndoverlay(self):
        specs = [(0, ('Points', 'Points', '', 0)), (1, ('Points', 'Points', '', 1)), (2, ('Points', 'Points', '', 2))]
        spec = ('Scatter', 'Points', '', 5)
        self.assertEqual(closest_match(spec, specs), None)
        spec = ('Scatter', 'Bar', 'Foo', 5)
        self.assertEqual(closest_match(spec, specs), None)
        spec = ('Scatter', 'Foo', 'Bar', 5)
        self.assertEqual(closest_match(spec, specs), None)
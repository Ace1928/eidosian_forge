import datetime as dt
from unittest import SkipTest, skipIf
import colorcet as cc
import numpy as np
import pandas as pd
import pytest
from numpy import nan
from packaging.version import Version
from holoviews import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation import apply_when
from holoviews.streams import Tap
from holoviews.util import render
import logging
class DatashaderStackTests(ComparisonTestCase):

    def setUp(self):
        self.rgb1_arr = np.array([[[0, 1], [1, 0]], [[1, 0], [0, 1]], [[0, 0], [0, 0]]], dtype=np.uint8).T * 255
        self.rgb2_arr = np.array([[[0, 0], [0, 0]], [[0, 0], [0, 0]], [[1, 0], [0, 1]]], dtype=np.uint8).T * 255
        self.rgb1 = RGB(self.rgb1_arr)
        self.rgb2 = RGB(self.rgb2_arr)

    def test_stack_add_compositor(self):
        combined = stack(self.rgb1 * self.rgb2, compositor='add')
        arr = np.array([[[0, 255, 255], [255, 0, 0]], [[255, 0, 0], [0, 255, 255]]], dtype=np.uint8)
        expected = RGB(arr)
        self.assertEqual(combined, expected)

    def test_stack_over_compositor(self):
        combined = stack(self.rgb1 * self.rgb2, compositor='over')
        self.assertEqual(combined, self.rgb2)

    def test_stack_over_compositor_reverse(self):
        combined = stack(self.rgb2 * self.rgb1, compositor='over')
        self.assertEqual(combined, self.rgb1)

    def test_stack_saturate_compositor(self):
        combined = stack(self.rgb1 * self.rgb2, compositor='saturate')
        self.assertEqual(combined, self.rgb1)

    def test_stack_saturate_compositor_reverse(self):
        combined = stack(self.rgb2 * self.rgb1, compositor='saturate')
        self.assertEqual(combined, self.rgb2)
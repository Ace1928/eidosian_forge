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
class DatashaderRegridTests(ComparisonTestCase):
    """
    Tests for datashader aggregation
    """

    def setUp(self):
        if ds_version <= Version('0.5.0'):
            raise SkipTest('Regridding operations require datashader>=0.6.0')

    def test_regrid_mean(self):
        img = Image((range(10), range(5), np.arange(10) * np.arange(5)[np.newaxis].T))
        regridded = regrid(img, width=2, height=2, dynamic=False)
        expected = Image(([2.0, 7.0], [0.75, 3.25], [[1, 5], [6, 22]]))
        self.assertEqual(regridded, expected)

    def test_regrid_mean_xarray_transposed(self):
        img = Image((range(10), range(5), np.arange(10) * np.arange(5)[np.newaxis].T), datatype=['xarray'])
        img.data = img.data.transpose()
        regridded = regrid(img, width=2, height=2, dynamic=False)
        expected = Image(([2.0, 7.0], [0.75, 3.25], [[1, 5], [6, 22]]))
        self.assertEqual(regridded, expected)

    def test_regrid_rgb_mean(self):
        arr = (np.arange(10) * np.arange(5)[np.newaxis].T).astype('float64')
        rgb = RGB((range(10), range(5), arr, arr * 2, arr * 2))
        regridded = regrid(rgb, width=2, height=2, dynamic=False)
        new_arr = np.array([[1.6, 5.6], [6.4, 22.4]])
        expected = RGB(([2.0, 7.0], [0.75, 3.25], new_arr, new_arr * 2, new_arr * 2), datatype=['xarray'])
        self.assertEqual(regridded, expected)

    def test_regrid_max(self):
        img = Image((range(10), range(5), np.arange(10) * np.arange(5)[np.newaxis].T))
        regridded = regrid(img, aggregator='max', width=2, height=2, dynamic=False)
        expected = Image(([2.0, 7.0], [0.75, 3.25], [[8, 18], [16, 36]]))
        self.assertEqual(regridded, expected)

    def test_regrid_upsampling(self):
        img = Image(([0.5, 1.5], [0.5, 1.5], [[0, 1], [2, 3]]))
        regridded = regrid(img, width=4, height=4, upsample=True, dynamic=False)
        expected = Image(([0.25, 0.75, 1.25, 1.75], [0.25, 0.75, 1.25, 1.75], [[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 3, 3], [2, 2, 3, 3]]))
        self.assertEqual(regridded, expected)

    def test_regrid_upsampling_linear(self):
        img = Image(([0.5, 1.5], [0.5, 1.5], [[0, 1], [2, 3]]))
        regridded = regrid(img, width=4, height=4, upsample=True, interpolation='linear', dynamic=False)
        expected = Image(([0.25, 0.75, 1.25, 1.75], [0.25, 0.75, 1.25, 1.75], [[0, 0, 0, 1], [0, 1, 1, 1], [1, 1, 2, 2], [2, 2, 2, 3]]))
        self.assertEqual(regridded, expected)

    def test_regrid_disabled_upsampling(self):
        img = Image(([0.5, 1.5], [0.5, 1.5], [[0, 1], [2, 3]]))
        regridded = regrid(img, width=3, height=3, dynamic=False, upsample=False)
        self.assertEqual(regridded, img)

    def test_regrid_disabled_expand(self):
        img = Image(([0.5, 1.5], [0.5, 1.5], [[0.0, 1.0], [2.0, 3.0]]))
        regridded = regrid(img, width=2, height=2, x_range=(-2, 4), y_range=(-2, 4), expand=False, dynamic=False)
        self.assertEqual(regridded, img)

    def test_regrid_zero_range(self):
        ls = np.linspace(0, 10, 200)
        xx, yy = np.meshgrid(ls, ls)
        img = Image(np.sin(xx) * np.cos(yy), bounds=(0, 0, 1, 1))
        regridded = regrid(img, x_range=(-1, -0.5), y_range=(-1, -0.5), dynamic=False)
        expected = Image(np.zeros((0, 0)), bounds=(0, 0, 0, 0), xdensity=1, ydensity=1)
        self.assertEqual(regridded, expected)
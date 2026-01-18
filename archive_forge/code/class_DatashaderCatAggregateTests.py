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
class DatashaderCatAggregateTests(ComparisonTestCase):

    def setUp(self):
        if ds_version < Version('0.11.0'):
            raise SkipTest('Regridding operations require datashader>=0.11.0')

    def test_aggregate_points_categorical(self):
        points = Points([(0.2, 0.3, 'A'), (0.4, 0.7, 'B'), (0, 0.99, 'C')], vdims='z')
        img = aggregate(points, dynamic=False, x_range=(0, 1), y_range=(0, 1), width=2, height=2, aggregator=ds.by('z', ds.count()))
        x = np.array([0.25, 0.75])
        y = np.array([0.25, 0.75])
        a = np.array([[1, 0], [0, 0]])
        b = np.array([[0, 1], [0, 0]])
        c = np.array([[0, 1], [0, 0]])
        xrds = xr.Dataset(coords={'x': x, 'y': y}, data_vars={'a': (('x', 'y'), a), 'b': (('x', 'y'), b), 'c': (('x', 'y'), c)})
        expected = ImageStack(xrds, kdims=['x', 'y'], vdims=['a', 'b', 'c'])
        actual = img.data
        assert (expected.data.to_array('z').values == actual.T.values).all()

    def test_aggregate_points_categorical_mean(self):
        points = Points([(0.2, 0.3, 'A', 0.1), (0.4, 0.7, 'B', 0.2), (0, 0.99, 'C', 0.3)], vdims=['cat', 'z'])
        img = aggregate(points, dynamic=False, x_range=(0, 1), y_range=(0, 1), width=2, height=2, aggregator=ds.by('cat', ds.mean('z')))
        x = np.array([0.25, 0.75])
        y = np.array([0.25, 0.75])
        a = np.array([[0.1, np.nan], [np.nan, np.nan]])
        b = np.array([[np.nan, 0.2], [np.nan, np.nan]])
        c = np.array([[np.nan, 0.3], [np.nan, np.nan]])
        xrds = xr.Dataset(coords={'x': x, 'y': y}, data_vars={'a': (('x', 'y'), a), 'b': (('x', 'y'), b), 'c': (('x', 'y'), c)})
        expected = ImageStack(xrds, kdims=['x', 'y'], vdims=['a', 'b', 'c'])
        actual = img.data
        np.testing.assert_equal(expected.data.to_array('z').values, actual.T.values)
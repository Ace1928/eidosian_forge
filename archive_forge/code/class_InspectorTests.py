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
class InspectorTests(ComparisonTestCase):
    """
    Tests for inspector operations
    """

    def setUp(self):
        points = Points([(0.2, 0.3), (0.4, 0.7), (0, 0.99)])
        self.pntsimg = rasterize(points, dynamic=False, x_range=(0, 1), y_range=(0, 1), width=4, height=4)
        if spatialpandas is None:
            return
        xs1, xs2, ys1, ys2 = ([1, 2, 3], [6, 7, 3], [2, 0, 7], [7, 5, 2])
        holes = [[[(1.5, 2), (2, 3), (1.6, 1.6)], [(2.1, 4.5), (2.5, 5), (2.3, 3.5)]]]
        polydata = [{'x': xs1, 'y': ys1, 'holes': holes, 'z': 1}, {'x': xs2, 'y': ys2, 'holes': [[]], 'z': 2}]
        self.polysrgb = datashade(Polygons(polydata, vdims=['z'], datatype=['spatialpandas']), x_range=(0, 7), y_range=(0, 7), dynamic=False)

    def tearDown(self):
        Tap.x, Tap.y = (None, None)

    def test_inspect_points_or_polygons(self):
        if spatialpandas is None:
            raise SkipTest('Polygon inspect tests require spatialpandas')
        polys = inspect(self.polysrgb, max_indicators=3, dynamic=False, pixels=1, x=6, y=5)
        self.assertEqual(polys, Polygons([{'x': [6, 3, 7], 'y': [7, 2, 5], 'z': 2}], vdims='z'))
        points = inspect(self.pntsimg, max_indicators=3, dynamic=False, pixels=1, x=-0.1, y=-0.1)
        self.assertEqual(points.dimension_values('x'), np.array([]))
        self.assertEqual(points.dimension_values('y'), np.array([]))

    def test_points_inspection_1px_mask(self):
        points = inspect_points(self.pntsimg, max_indicators=3, dynamic=False, pixels=1, x=-0.1, y=-0.1)
        self.assertEqual(points.dimension_values('x'), np.array([]))
        self.assertEqual(points.dimension_values('y'), np.array([]))

    def test_points_inspection_2px_mask(self):
        points = inspect_points(self.pntsimg, max_indicators=3, dynamic=False, pixels=2, x=-0.1, y=-0.1)
        self.assertEqual(points.dimension_values('x'), np.array([0.2]))
        self.assertEqual(points.dimension_values('y'), np.array([0.3]))

    def test_points_inspection_4px_mask(self):
        points = inspect_points(self.pntsimg, max_indicators=3, dynamic=False, pixels=4, x=-0.1, y=-0.1)
        self.assertEqual(points.dimension_values('x'), np.array([0.2, 0.4]))
        self.assertEqual(points.dimension_values('y'), np.array([0.3, 0.7]))

    def test_points_inspection_5px_mask(self):
        points = inspect_points(self.pntsimg, max_indicators=3, dynamic=False, pixels=5, x=-0.1, y=-0.1)
        self.assertEqual(points.dimension_values('x'), np.array([0.2, 0.4, 0]))
        self.assertEqual(points.dimension_values('y'), np.array([0.3, 0.7, 0.99]))

    def test_inspection_5px_mask_points_df(self):
        inspector = inspect.instance(max_indicators=3, dynamic=False, pixels=5, x=-0.1, y=-0.1)
        inspector(self.pntsimg)
        self.assertEqual(list(inspector.hits['x']), [0.2, 0.4, 0.0])
        self.assertEqual(list(inspector.hits['y']), [0.3, 0.7, 0.99])

    def test_points_inspection_dict_streams(self):
        Tap.x, Tap.y = (0.4, 0.7)
        points = inspect_points(self.pntsimg, max_indicators=3, dynamic=True, pixels=1, streams=dict(x=Tap.param.x, y=Tap.param.y))
        self.assertEqual(len(points.streams), 1)
        self.assertEqual(isinstance(points.streams[0], Tap), True)
        self.assertEqual(points.streams[0].x, 0.4)
        self.assertEqual(points.streams[0].y, 0.7)

    def test_points_inspection_dict_streams_instance(self):
        Tap.x, Tap.y = (0.2, 0.3)
        inspector = inspect_points.instance(max_indicators=3, dynamic=True, pixels=1, streams=dict(x=Tap.param.x, y=Tap.param.y))
        points = inspector(self.pntsimg)
        self.assertEqual(len(points.streams), 1)
        self.assertEqual(isinstance(points.streams[0], Tap), True)
        self.assertEqual(points.streams[0].x, 0.2)
        self.assertEqual(points.streams[0].y, 0.3)

    def test_polys_inspection_1px_mask_hit(self):
        if spatialpandas is None:
            raise SkipTest('Polygon inspect tests require spatialpandas')
        polys = inspect_polygons(self.polysrgb, max_indicators=3, dynamic=False, pixels=1, x=6, y=5)
        self.assertEqual(polys, Polygons([{'x': [6, 3, 7], 'y': [7, 2, 5], 'z': 2}], vdims='z'))

    def test_inspection_1px_mask_poly_df(self):
        if spatialpandas is None:
            raise SkipTest('Polygon inspect tests require spatialpandas')
        inspector = inspect.instance(max_indicators=3, dynamic=False, pixels=1, x=6, y=5)
        inspector(self.polysrgb)
        self.assertEqual(len(inspector.hits), 1)
        data = [[6.0, 7.0, 3.0, 2.0, 7.0, 5.0, 6.0, 7.0]]
        self.assertEqual(inspector.hits.iloc[0].geometry, spatialpandas.geometry.polygon.Polygon(data))

    def test_polys_inspection_1px_mask_miss(self):
        if spatialpandas is None:
            raise SkipTest('Polygon inspect tests require spatialpandas')
        polys = inspect_polygons(self.polysrgb, max_indicators=3, dynamic=False, pixels=1, x=0, y=0)
        self.assertEqual(polys, Polygons([], vdims='z'))
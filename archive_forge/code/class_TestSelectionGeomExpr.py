from unittest import skipIf
import numpy as np
import pandas as pd
import pytest
from holoviews.core import NdOverlay
from holoviews.core.options import Store
from holoviews.element import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.element.selection import spatial_select_columnar
class TestSelectionGeomExpr(ComparisonTestCase):

    def setUp(self):
        import holoviews.plotting.bokeh
        super().setUp()
        self._backend = Store.current_backend
        Store.set_current_backend('bokeh')

    def tearDown(self):
        Store.current_backend = self._backend

    def test_rect_selection_numeric(self):
        rect = Rectangles([(0, 1, 2, 3), (1, 3, 1.5, 4), (2.5, 4.2, 3.5, 4.8)])
        expr, bbox, region = rect._get_selection_expr_for_stream_value(bounds=(0.5, 0.9, 3.4, 4.9))
        self.assertEqual(bbox, {'x0': (0.5, 3.4), 'y0': (0.9, 4.9), 'x1': (0.5, 3.4), 'y1': (0.9, 4.9)})
        self.assertEqual(expr.apply(rect), np.array([False, True, False]))
        self.assertEqual(region, Rectangles([(0.5, 0.9, 3.4, 4.9)]) * Path([]))
        expr, bbox, region = rect._get_selection_expr_for_stream_value(bounds=(0, 0.9, 3.5, 4.9))
        self.assertEqual(bbox, {'x0': (0, 3.5), 'y0': (0.9, 4.9), 'x1': (0, 3.5), 'y1': (0.9, 4.9)})
        self.assertEqual(expr.apply(rect), np.array([True, True, True]))
        self.assertEqual(region, Rectangles([(0, 0.9, 3.5, 4.9)]) * Path([]))

    def test_rect_selection_numeric_inverted(self):
        rect = Rectangles([(0, 1, 2, 3), (1, 3, 1.5, 4), (2.5, 4.2, 3.5, 4.8)]).opts(invert_axes=True)
        expr, bbox, region = rect._get_selection_expr_for_stream_value(bounds=(0.9, 0.5, 4.9, 3.4))
        self.assertEqual(bbox, {'x0': (0.5, 3.4), 'y0': (0.9, 4.9), 'x1': (0.5, 3.4), 'y1': (0.9, 4.9)})
        self.assertEqual(expr.apply(rect), np.array([False, True, False]))
        self.assertEqual(region, Rectangles([(0.9, 0.5, 4.9, 3.4)]) * Path([]))
        expr, bbox, region = rect._get_selection_expr_for_stream_value(bounds=(0.9, 0, 4.9, 3.5))
        self.assertEqual(bbox, {'x0': (0, 3.5), 'y0': (0.9, 4.9), 'x1': (0, 3.5), 'y1': (0.9, 4.9)})
        self.assertEqual(expr.apply(rect), np.array([True, True, True]))
        self.assertEqual(region, Rectangles([(0.9, 0, 4.9, 3.5)]) * Path([]))

    @shapely_available
    def test_rect_geom_selection(self):
        rect = Rectangles([(0, 1, 2, 3), (1, 3, 1.5, 4), (2.5, 4.2, 3.5, 4.8)])
        geom = np.array([(-0.4, -0.1), (2.2, -0.1), (2.2, 4.1), (-0.1, 4.2)])
        expr, bbox, region = rect._get_selection_expr_for_stream_value(geometry=geom)
        self.assertEqual(bbox, {'x0': np.array([-0.4, 2.2, 2.2, -0.1]), 'y0': np.array([-0.1, -0.1, 4.1, 4.2]), 'x1': np.array([-0.4, 2.2, 2.2, -0.1]), 'y1': np.array([-0.1, -0.1, 4.1, 4.2])})
        self.assertEqual(expr.apply(rect), np.array([True, True, False]))
        self.assertEqual(region, Rectangles([]) * Path([list(geom) + [(-0.4, -0.1)]]))

    @shapely_available
    def test_rect_geom_selection_inverted(self):
        rect = Rectangles([(0, 1, 2, 3), (1, 3, 1.5, 4), (2.5, 4.2, 3.5, 4.8)]).opts(invert_axes=True)
        geom = np.array([(-0.4, -0.1), (3.2, -0.1), (3.2, 4.1), (-0.1, 4.2)])
        expr, bbox, region = rect._get_selection_expr_for_stream_value(geometry=geom)
        self.assertEqual(bbox, {'y0': np.array([-0.4, 3.2, 3.2, -0.1]), 'x0': np.array([-0.1, -0.1, 4.1, 4.2]), 'y1': np.array([-0.4, 3.2, 3.2, -0.1]), 'x1': np.array([-0.1, -0.1, 4.1, 4.2])})
        self.assertEqual(expr.apply(rect), np.array([True, False, False]))
        self.assertEqual(region, Rectangles([]) * Path([list(geom) + [(-0.4, -0.1)]]))

    def test_segments_selection_numeric(self):
        segs = Segments([(0, 1, 2, 3), (1, 3, 1.5, 4), (2.5, 4.2, 3.5, 4.8)])
        expr, bbox, region = segs._get_selection_expr_for_stream_value(bounds=(0.5, 0.9, 3.4, 4.9))
        self.assertEqual(bbox, {'x0': (0.5, 3.4), 'y0': (0.9, 4.9), 'x1': (0.5, 3.4), 'y1': (0.9, 4.9)})
        self.assertEqual(expr.apply(segs), np.array([False, True, False]))
        self.assertEqual(region, Rectangles([(0.5, 0.9, 3.4, 4.9)]) * Path([]))
        expr, bbox, region = segs._get_selection_expr_for_stream_value(bounds=(0, 0.9, 3.5, 4.9))
        self.assertEqual(bbox, {'x0': (0, 3.5), 'y0': (0.9, 4.9), 'x1': (0, 3.5), 'y1': (0.9, 4.9)})
        self.assertEqual(expr.apply(segs), np.array([True, True, True]))
        self.assertEqual(region, Rectangles([(0, 0.9, 3.5, 4.9)]) * Path([]))

    def test_segs_selection_numeric_inverted(self):
        segs = Segments([(0, 1, 2, 3), (1, 3, 1.5, 4), (2.5, 4.2, 3.5, 4.8)]).opts(invert_axes=True)
        expr, bbox, region = segs._get_selection_expr_for_stream_value(bounds=(0.9, 0.5, 4.9, 3.4))
        self.assertEqual(bbox, {'x0': (0.5, 3.4), 'y0': (0.9, 4.9), 'x1': (0.5, 3.4), 'y1': (0.9, 4.9)})
        self.assertEqual(expr.apply(segs), np.array([False, True, False]))
        self.assertEqual(region, Rectangles([(0.9, 0.5, 4.9, 3.4)]) * Path([]))
        expr, bbox, region = segs._get_selection_expr_for_stream_value(bounds=(0.9, 0, 4.9, 3.5))
        self.assertEqual(bbox, {'x0': (0, 3.5), 'y0': (0.9, 4.9), 'x1': (0, 3.5), 'y1': (0.9, 4.9)})
        self.assertEqual(expr.apply(segs), np.array([True, True, True]))
        self.assertEqual(region, Rectangles([(0.9, 0, 4.9, 3.5)]) * Path([]))

    @shapely_available
    def test_segs_geom_selection(self):
        rect = Segments([(0, 1, 2, 3), (1, 3, 1.5, 4), (2.5, 4.2, 3.5, 4.8)])
        geom = np.array([(-0.4, -0.1), (2.2, -0.1), (2.2, 4.1), (-0.1, 4.2)])
        expr, bbox, region = rect._get_selection_expr_for_stream_value(geometry=geom)
        self.assertEqual(bbox, {'x0': np.array([-0.4, 2.2, 2.2, -0.1]), 'y0': np.array([-0.1, -0.1, 4.1, 4.2]), 'x1': np.array([-0.4, 2.2, 2.2, -0.1]), 'y1': np.array([-0.1, -0.1, 4.1, 4.2])})
        self.assertEqual(expr.apply(rect), np.array([True, True, False]))
        self.assertEqual(region, Rectangles([]) * Path([list(geom) + [(-0.4, -0.1)]]))

    @shapely_available
    def test_segs_geom_selection_inverted(self):
        rect = Segments([(0, 1, 2, 3), (1, 3, 1.5, 4), (2.5, 4.2, 3.5, 4.8)]).opts(invert_axes=True)
        geom = np.array([(-0.4, -0.1), (3.2, -0.1), (3.2, 4.1), (-0.1, 4.2)])
        expr, bbox, region = rect._get_selection_expr_for_stream_value(geometry=geom)
        self.assertEqual(bbox, {'y0': np.array([-0.4, 3.2, 3.2, -0.1]), 'x0': np.array([-0.1, -0.1, 4.1, 4.2]), 'y1': np.array([-0.4, 3.2, 3.2, -0.1]), 'x1': np.array([-0.1, -0.1, 4.1, 4.2])})
        self.assertEqual(expr.apply(rect), np.array([True, False, False]))
        self.assertEqual(region, Rectangles([]) * Path([list(geom) + [(-0.4, -0.1)]]))
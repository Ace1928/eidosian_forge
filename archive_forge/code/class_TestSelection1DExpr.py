from unittest import skipIf
import numpy as np
import pandas as pd
import pytest
from holoviews.core import NdOverlay
from holoviews.core.options import Store
from holoviews.element import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.element.selection import spatial_select_columnar
class TestSelection1DExpr(ComparisonTestCase):

    def setUp(self):
        import holoviews.plotting.bokeh
        super().setUp()
        self._backend = Store.current_backend
        Store.set_current_backend('bokeh')

    def tearDown(self):
        Store.current_backend = self._backend

    def test_area_selection_numeric(self):
        area = Area([3, 2, 1, 3, 4])
        expr, bbox, region = area._get_selection_expr_for_stream_value(bounds=(1, 0, 3, 2))
        self.assertEqual(bbox, {'x': (1, 3)})
        self.assertEqual(expr.apply(area), np.array([False, True, True, True, False]))
        self.assertEqual(region, NdOverlay({0: VSpan(1, 3)}))

    def test_area_selection_numeric_inverted(self):
        area = Area([3, 2, 1, 3, 4]).opts(invert_axes=True)
        expr, bbox, region = area._get_selection_expr_for_stream_value(bounds=(0, 1, 2, 3))
        self.assertEqual(bbox, {'x': (1, 3)})
        self.assertEqual(expr.apply(area), np.array([False, True, True, True, False]))
        self.assertEqual(region, NdOverlay({0: HSpan(1, 3)}))

    def test_area_selection_categorical(self):
        area = Area((['B', 'A', 'C', 'D', 'E'], [3, 2, 1, 3, 4]))
        expr, bbox, region = area._get_selection_expr_for_stream_value(bounds=(0, 1, 2, 3), x_selection=['B', 'A', 'C'])
        self.assertEqual(bbox, {'x': ['B', 'A', 'C']})
        self.assertEqual(expr.apply(area), np.array([True, True, True, False, False]))
        self.assertEqual(region, NdOverlay({0: VSpan(0, 2)}))

    def test_area_selection_numeric_index_cols(self):
        area = Area([3, 2, 1, 3, 2])
        expr, bbox, region = area._get_selection_expr_for_stream_value(bounds=(1, 0, 3, 2), index_cols=['y'])
        self.assertEqual(bbox, {'x': (1, 3)})
        self.assertEqual(expr.apply(area), np.array([False, True, True, False, True]))
        self.assertEqual(region, None)

    def test_curve_selection_numeric(self):
        curve = Curve([3, 2, 1, 3, 4])
        expr, bbox, region = curve._get_selection_expr_for_stream_value(bounds=(1, 0, 3, 2))
        self.assertEqual(bbox, {'x': (1, 3)})
        self.assertEqual(expr.apply(curve), np.array([False, True, True, True, False]))
        self.assertEqual(region, NdOverlay({0: VSpan(1, 3)}))

    def test_curve_selection_categorical(self):
        curve = Curve((['B', 'A', 'C', 'D', 'E'], [3, 2, 1, 3, 4]))
        expr, bbox, region = curve._get_selection_expr_for_stream_value(bounds=(0, 1, 2, 3), x_selection=['B', 'A', 'C'])
        self.assertEqual(bbox, {'x': ['B', 'A', 'C']})
        self.assertEqual(expr.apply(curve), np.array([True, True, True, False, False]))
        self.assertEqual(region, NdOverlay({0: VSpan(0, 2)}))

    def test_curve_selection_numeric_index_cols(self):
        curve = Curve([3, 2, 1, 3, 2])
        expr, bbox, region = curve._get_selection_expr_for_stream_value(bounds=(1, 0, 3, 2), index_cols=['y'])
        self.assertEqual(bbox, {'x': (1, 3)})
        self.assertEqual(expr.apply(curve), np.array([False, True, True, False, True]))
        self.assertEqual(region, None)

    def test_box_whisker_single(self):
        box_whisker = BoxWhisker(list(range(10)))
        expr, bbox, region = box_whisker._get_selection_expr_for_stream_value(bounds=(0, 3, 1, 7))
        self.assertEqual(bbox, {'y': (3, 7)})
        self.assertEqual(expr.apply(box_whisker), np.array([False, False, False, True, True, True, True, True, False, False]))
        self.assertEqual(region, NdOverlay({0: HSpan(3, 7)}))

    def test_box_whisker_single_inverted(self):
        box = BoxWhisker(list(range(10))).opts(invert_axes=True)
        expr, bbox, region = box._get_selection_expr_for_stream_value(bounds=(3, 0, 7, 1))
        self.assertEqual(bbox, {'y': (3, 7)})
        self.assertEqual(expr.apply(box), np.array([False, False, False, True, True, True, True, True, False, False]))
        self.assertEqual(region, NdOverlay({0: VSpan(3, 7)}))

    def test_box_whisker_cats(self):
        box_whisker = BoxWhisker((['A', 'A', 'A', 'B', 'B', 'C', 'C', 'C', 'C', 'C'], list(range(10))), 'x', 'y')
        expr, bbox, region = box_whisker._get_selection_expr_for_stream_value(bounds=(0, 1, 2, 7), x_selection=['A', 'B'])
        self.assertEqual(bbox, {'y': (1, 7), 'x': ['A', 'B']})
        self.assertEqual(expr.apply(box_whisker), np.array([False, True, True, True, True, False, False, False, False, False]))
        self.assertEqual(region, NdOverlay({0: HSpan(1, 7)}))

    def test_box_whisker_cats_index_cols(self):
        box_whisker = BoxWhisker((['A', 'A', 'A', 'B', 'B', 'C', 'C', 'C', 'C', 'C'], list(range(10))), 'x', 'y')
        expr, bbox, region = box_whisker._get_selection_expr_for_stream_value(bounds=(0, 1, 2, 7), x_selection=['A', 'B'], index_cols=['x'])
        self.assertEqual(bbox, {'y': (1, 7), 'x': ['A', 'B']})
        self.assertEqual(expr.apply(box_whisker), np.array([True, True, True, True, True, False, False, False, False, False]))
        self.assertEqual(region, None)

    def test_violin_single(self):
        violin = Violin(list(range(10)))
        expr, bbox, region = violin._get_selection_expr_for_stream_value(bounds=(0, 3, 1, 7))
        self.assertEqual(bbox, {'y': (3, 7)})
        self.assertEqual(expr.apply(violin), np.array([False, False, False, True, True, True, True, True, False, False]))
        self.assertEqual(region, NdOverlay({0: HSpan(3, 7)}))

    def test_violin_single_inverted(self):
        violin = Violin(list(range(10))).opts(invert_axes=True)
        expr, bbox, region = violin._get_selection_expr_for_stream_value(bounds=(3, 0, 7, 1))
        self.assertEqual(bbox, {'y': (3, 7)})
        self.assertEqual(expr.apply(violin), np.array([False, False, False, True, True, True, True, True, False, False]))
        self.assertEqual(region, NdOverlay({0: VSpan(3, 7)}))

    def test_violin_cats(self):
        violin = Violin((['A', 'A', 'A', 'B', 'B', 'C', 'C', 'C', 'C', 'C'], list(range(10))), 'x', 'y')
        expr, bbox, region = violin._get_selection_expr_for_stream_value(bounds=(0, 1, 2, 7), x_selection=['A', 'B'])
        self.assertEqual(bbox, {'y': (1, 7), 'x': ['A', 'B']})
        self.assertEqual(expr.apply(violin), np.array([False, True, True, True, True, False, False, False, False, False]))
        self.assertEqual(region, NdOverlay({0: HSpan(1, 7)}))

    def test_violin_cats_index_cols(self):
        violin = Violin((['A', 'A', 'A', 'B', 'B', 'C', 'C', 'C', 'C', 'C'], list(range(10))), 'x', 'y')
        expr, bbox, region = violin._get_selection_expr_for_stream_value(bounds=(0, 1, 2, 7), x_selection=['A', 'B'], index_cols=['x'])
        self.assertEqual(bbox, {'y': (1, 7), 'x': ['A', 'B']})
        self.assertEqual(expr.apply(violin), np.array([True, True, True, True, True, False, False, False, False, False]))
        self.assertEqual(region, None)

    def test_distribution_single(self):
        dist = Distribution(list(range(10)))
        expr, bbox, region = dist._get_selection_expr_for_stream_value(bounds=(3, 0, 7, 1))
        self.assertEqual(bbox, {'Value': (3, 7)})
        self.assertEqual(expr.apply(dist), np.array([False, False, False, True, True, True, True, True, False, False]))
        self.assertEqual(region, NdOverlay({0: VSpan(3, 7)}))

    def test_distribution_single_inverted(self):
        dist = Distribution(list(range(10))).opts(invert_axes=True)
        expr, bbox, region = dist._get_selection_expr_for_stream_value(bounds=(0, 3, 1, 7))
        self.assertEqual(bbox, {'Value': (3, 7)})
        self.assertEqual(expr.apply(dist), np.array([False, False, False, True, True, True, True, True, False, False]))
        self.assertEqual(region, NdOverlay({0: HSpan(3, 7)}))
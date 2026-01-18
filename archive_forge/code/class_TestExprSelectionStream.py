from collections import defaultdict
from unittest import SkipTest
import pandas as pd
import param
import pytest
from panel.widgets import IntSlider
import holoviews as hv
from holoviews.core.spaces import DynamicMap
from holoviews.core.util import Version
from holoviews.element import Curve, Histogram, Points, Polygons, Scatter
from holoviews.element.comparison import ComparisonTestCase
from holoviews.streams import *  # noqa (Test all available streams)
from holoviews.util import Dynamic, extension
from holoviews.util.transform import dim
from .utils import LoggingComparisonTestCase
class TestExprSelectionStream(ComparisonTestCase):

    def setUp(self):
        extension('bokeh')

    def test_selection_expr_stream_2D_elements(self):
        element_type_2D = [Points]
        for element_type in element_type_2D:
            element = element_type(([1, 2, 3], [1, 5, 10]))
            expr_stream = SelectionExpr(element)
            self.assertEqual(len(expr_stream.input_streams), 3)
            self.assertIsInstance(expr_stream.input_streams[0], SelectionXY)
            self.assertIsInstance(expr_stream.input_streams[1], Lasso)
            self.assertIsInstance(expr_stream.input_streams[2], Selection1D)
            self.assertIsNone(expr_stream.bbox)
            self.assertIsNone(expr_stream.selection_expr)
            expr_stream.input_streams[0].event(bounds=(1, 1, 3, 4))
            self.assertEqual(repr(expr_stream.selection_expr), repr((dim('x') >= 1) & (dim('x') <= 3) & ((dim('y') >= 1) & (dim('y') <= 4))))
            self.assertEqual(expr_stream.bbox, {'x': (1, 3), 'y': (1, 4)})

    def test_selection_expr_stream_1D_elements(self):
        element_type_1D = [Scatter]
        for element_type in element_type_1D:
            element = element_type(([1, 2, 3], [1, 5, 10]))
            expr_stream = SelectionExpr(element)
            self.assertEqual(len(expr_stream.input_streams), 1)
            self.assertIsInstance(expr_stream.input_streams[0], SelectionXY)
            self.assertIsNone(expr_stream.bbox)
            self.assertIsNone(expr_stream.selection_expr)
            expr_stream.input_streams[0].event(bounds=(1, 1, 3, 4))
            self.assertEqual(repr(expr_stream.selection_expr), repr((dim('x') >= 1) & (dim('x') <= 3)))
            self.assertEqual(expr_stream.bbox, {'x': (1, 3)})

    def test_selection_expr_stream_invert_axes_2D_elements(self):
        element_type_2D = [Points]
        for element_type in element_type_2D:
            element = element_type(([1, 2, 3], [1, 5, 10])).opts(invert_axes=True)
            expr_stream = SelectionExpr(element)
            self.assertEqual(len(expr_stream.input_streams), 3)
            self.assertIsInstance(expr_stream.input_streams[0], SelectionXY)
            self.assertIsInstance(expr_stream.input_streams[1], Lasso)
            self.assertIsInstance(expr_stream.input_streams[2], Selection1D)
            self.assertIsNone(expr_stream.bbox)
            self.assertIsNone(expr_stream.selection_expr)
            expr_stream.input_streams[0].event(bounds=(1, 1, 3, 4))
            self.assertEqual(repr(expr_stream.selection_expr), repr((dim('y') >= 1) & (dim('y') <= 3) & ((dim('x') >= 1) & (dim('x') <= 4))))
            self.assertEqual(expr_stream.bbox, {'y': (1, 3), 'x': (1, 4)})

    def test_selection_expr_stream_invert_axes_1D_elements(self):
        element_type_1D = [Scatter]
        for element_type in element_type_1D:
            element = element_type(([1, 2, 3], [1, 5, 10])).opts(invert_axes=True)
            expr_stream = SelectionExpr(element)
            self.assertEqual(len(expr_stream.input_streams), 1)
            self.assertIsInstance(expr_stream.input_streams[0], SelectionXY)
            self.assertIsNone(expr_stream.bbox)
            self.assertIsNone(expr_stream.selection_expr)
            expr_stream.input_streams[0].event(bounds=(1, 1, 3, 4))
            self.assertEqual(repr(expr_stream.selection_expr), repr((dim('x') >= 1) & (dim('x') <= 4)))
            self.assertEqual(expr_stream.bbox, {'x': (1, 4)})

    def test_selection_expr_stream_invert_xaxis_yaxis_2D_elements(self):
        element_type_2D = [Points]
        for element_type in element_type_2D:
            element = element_type(([1, 2, 3], [1, 5, 10])).opts(invert_xaxis=True, invert_yaxis=True)
            expr_stream = SelectionExpr(element)
            self.assertEqual(len(expr_stream.input_streams), 3)
            self.assertIsInstance(expr_stream.input_streams[0], SelectionXY)
            self.assertIsInstance(expr_stream.input_streams[1], Lasso)
            self.assertIsInstance(expr_stream.input_streams[2], Selection1D)
            self.assertIsNone(expr_stream.bbox)
            self.assertIsNone(expr_stream.selection_expr)
            expr_stream.input_streams[0].event(bounds=(3, 4, 1, 1))
            self.assertEqual(repr(expr_stream.selection_expr), repr((dim('x') >= 1) & (dim('x') <= 3) & ((dim('y') >= 1) & (dim('y') <= 4))))
            self.assertEqual(expr_stream.bbox, {'x': (1, 3), 'y': (1, 4)})

    def test_selection_expr_stream_invert_xaxis_yaxis_1D_elements(self):
        element_type_1D = [Scatter]
        for element_type in element_type_1D:
            element = element_type(([1, 2, 3], [1, 5, 10])).opts(invert_xaxis=True, invert_yaxis=True)
            expr_stream = SelectionExpr(element)
            self.assertEqual(len(expr_stream.input_streams), 1)
            self.assertIsInstance(expr_stream.input_streams[0], SelectionXY)
            self.assertIsNone(expr_stream.bbox)
            self.assertIsNone(expr_stream.selection_expr)
            expr_stream.input_streams[0].event(bounds=(3, 4, 1, 1))
            self.assertEqual(repr(expr_stream.selection_expr), repr((dim('x') >= 1) & (dim('x') <= 3)))
            self.assertEqual(expr_stream.bbox, {'x': (1, 3)})

    def test_selection_expr_stream_hist(self):
        hist = Histogram(([1, 2, 3, 4, 5], [1, 5, 2, 3, 7]))
        expr_stream = SelectionExpr(hist)
        self.assertEqual(len(expr_stream.input_streams), 1)
        self.assertIsInstance(expr_stream.input_streams[0], SelectionXY)
        self.assertIsNone(expr_stream.bbox)
        self.assertIsNone(expr_stream.selection_expr)
        expr_stream.input_streams[0].event(bounds=(1.5, 2.5, 4.6, 6))
        self.assertEqual(repr(expr_stream.selection_expr), repr((dim('x') >= 1.5) & (dim('x') <= 4.6)))
        self.assertEqual(expr_stream.bbox, {'x': (1.5, 4.6)})
        expr_stream.input_streams[0].event(bounds=(2.5, -10, 8, 10))
        self.assertEqual(repr(expr_stream.selection_expr), repr((dim('x') >= 2.5) & (dim('x') <= 8)))
        self.assertEqual(expr_stream.bbox, {'x': (2.5, 8)})

    def test_selection_expr_stream_hist_invert_axes(self):
        hist = Histogram(([1, 2, 3, 4, 5], [1, 5, 2, 3, 7])).opts(invert_axes=True)
        expr_stream = SelectionExpr(hist)
        self.assertEqual(len(expr_stream.input_streams), 1)
        self.assertIsInstance(expr_stream.input_streams[0], SelectionXY)
        self.assertIsNone(expr_stream.bbox)
        self.assertIsNone(expr_stream.selection_expr)
        expr_stream.input_streams[0].event(bounds=(2.5, 1.5, 6, 4.6))
        self.assertEqual(repr(expr_stream.selection_expr), repr((dim('x') >= 1.5) & (dim('x') <= 4.6)))
        self.assertEqual(expr_stream.bbox, {'x': (1.5, 4.6)})
        expr_stream.input_streams[0].event(bounds=(-10, 2.5, 10, 8))
        self.assertEqual(repr(expr_stream.selection_expr), repr((dim('x') >= 2.5) & (dim('x') <= 8)))
        self.assertEqual(expr_stream.bbox, {'x': (2.5, 8)})

    def test_selection_expr_stream_hist_invert_xaxis_yaxis(self):
        hist = Histogram(([1, 2, 3, 4, 5], [1, 5, 2, 3, 7])).opts(invert_xaxis=True, invert_yaxis=True)
        expr_stream = SelectionExpr(hist)
        self.assertEqual(len(expr_stream.input_streams), 1)
        self.assertIsInstance(expr_stream.input_streams[0], SelectionXY)
        self.assertIsNone(expr_stream.bbox)
        self.assertIsNone(expr_stream.selection_expr)
        expr_stream.input_streams[0].event(bounds=(4.6, 6, 1.5, 2.5))
        self.assertEqual(repr(expr_stream.selection_expr), repr((dim('x') >= 1.5) & (dim('x') <= 4.6)))
        self.assertEqual(expr_stream.bbox, {'x': (1.5, 4.6)})
        expr_stream.input_streams[0].event(bounds=(8, 10, 2.5, -10))
        self.assertEqual(repr(expr_stream.selection_expr), repr((dim('x') >= 2.5) & (dim('x') <= 8)))
        self.assertEqual(expr_stream.bbox, {'x': (2.5, 8)})

    def test_selection_expr_stream_polygon_index_cols(self):
        try:
            import shapely
        except ImportError:
            try:
                import spatialpandas
            except ImportError:
                raise SkipTest('Shapely required for polygon selection')
        poly = Polygons([[(0, 0, 'a'), (2, 0, 'a'), (1, 1, 'a')], [(2, 0, 'b'), (4, 0, 'b'), (3, 1, 'b')], [(1, 1, 'c'), (3, 1, 'c'), (2, 2, 'c')]], vdims=['cat'])
        events = []
        expr_stream = SelectionExpr(poly, index_cols=['cat'])
        expr_stream.add_subscriber(lambda **kwargs: events.append(kwargs))
        self.assertEqual(len(expr_stream.input_streams), 3)
        self.assertIsInstance(expr_stream.input_streams[0], SelectionXY)
        self.assertIsInstance(expr_stream.input_streams[1], Lasso)
        self.assertIsInstance(expr_stream.input_streams[2], Selection1D)
        self.assertIsNone(expr_stream.bbox)
        self.assertIsNone(expr_stream.selection_expr)
        expr_stream.input_streams[2].event(index=[0, 1])
        self.assertEqual(repr(expr_stream.selection_expr), repr(dim('cat').isin(['a', 'b'])))
        self.assertEqual(expr_stream.bbox, None)
        self.assertEqual(len(events), 1)
        expr_stream.input_streams[0].event(bounds=(0, 0, 4, 1))
        self.assertEqual(repr(expr_stream.selection_expr), repr(dim('cat').isin(['a', 'b'])))
        self.assertEqual(len(events), 1)
        expr_stream.input_streams[1].event(geometry=np.array([(0, 0), (4, 0), (4, 2), (0, 2)]))
        self.assertEqual(repr(expr_stream.selection_expr), repr(dim('cat').isin(['a', 'b', 'c'])))
        self.assertEqual(len(events), 2)
        expr_stream.input_streams[2].event(index=[1, 2])
        self.assertEqual(repr(expr_stream.selection_expr), repr(dim('cat').isin(['b', 'c'])))
        self.assertEqual(expr_stream.bbox, None)
        self.assertEqual(len(events), 3)

    def test_selection_expr_stream_dynamic_map_2D_elements(self):
        element_type_2D = [Points]
        for element_type in element_type_2D:
            dmap = Dynamic(element_type(([1, 2, 3], [1, 5, 10])))
            expr_stream = SelectionExpr(dmap)
            self.assertEqual(len(expr_stream.input_streams), 3)
            self.assertIsInstance(expr_stream.input_streams[0], SelectionXY)
            self.assertIsNone(expr_stream.bbox)
            self.assertIsNone(expr_stream.selection_expr)
            expr_stream.input_streams[0].event(bounds=(1, 1, 3, 4))
            self.assertEqual(repr(expr_stream.selection_expr), repr((dim('x') >= 1) & (dim('x') <= 3) & ((dim('y') >= 1) & (dim('y') <= 4))))
            self.assertEqual(expr_stream.bbox, {'x': (1, 3), 'y': (1, 4)})

    def test_selection_expr_stream_dynamic_map_1D_elements(self):
        element_type_1D = [Scatter]
        for element_type in element_type_1D:
            dmap = Dynamic(element_type(([1, 2, 3], [1, 5, 10])))
            expr_stream = SelectionExpr(dmap)
            self.assertEqual(len(expr_stream.input_streams), 1)
            self.assertIsInstance(expr_stream.input_streams[0], SelectionXY)
            self.assertIsNone(expr_stream.bbox)
            self.assertIsNone(expr_stream.selection_expr)
            expr_stream.input_streams[0].event(bounds=(1, 1, 3, 4))
            self.assertEqual(repr(expr_stream.selection_expr), repr((dim('x') >= 1) & (dim('x') <= 3)))
            self.assertEqual(expr_stream.bbox, {'x': (1, 3)})
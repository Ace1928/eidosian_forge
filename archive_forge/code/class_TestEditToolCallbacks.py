import datetime as dt
from collections import deque, namedtuple
from unittest import SkipTest
import numpy as np
import pandas as pd
import pytest
import pyviz_comms as comms
from bokeh.events import Tap
from bokeh.io.doc import set_curdoc
from bokeh.models import ColumnDataSource, Plot, PolyEditTool, Range1d, Selection
from holoviews.core import DynamicMap
from holoviews.core.options import Store
from holoviews.element import Box, Curve, Points, Polygons, Rectangles, Table
from holoviews.element.comparison import ComparisonTestCase
from holoviews.plotting.bokeh.callbacks import (
from holoviews.plotting.bokeh.renderer import BokehRenderer
from holoviews.streams import (
class TestEditToolCallbacks(CallbackTestCase):

    def test_point_draw_callback(self):
        points = Points([(0, 1)])
        point_draw = PointDraw(source=points)
        plot = bokeh_server_renderer.get_plot(points)
        self.assertIsInstance(plot.callbacks[0], PointDrawCallback)
        callback = plot.callbacks[0]
        data = {'x': [1, 2, 3], 'y': [1, 2, 3]}
        callback.on_msg({'data': data})
        self.assertEqual(point_draw.element, Points(data))

    def test_point_draw_callback_initialized_server(self):
        points = Points([(0, 1)])
        PointDraw(source=points)
        plot = bokeh_server_renderer.get_plot(points)
        assert 'data' in plot.handles['source']._callbacks

    def test_point_draw_callback_with_vdims_initialization(self):
        points = Points([(0, 1, 'A')], vdims=['A'])
        stream = PointDraw(source=points)
        bokeh_server_renderer.get_plot(points)
        self.assertEqual(stream.element.dimension_values('A'), np.array(['A']))

    def test_point_draw_callback_with_vdims(self):
        points = Points([(0, 1, 'A')], vdims=['A'])
        point_draw = PointDraw(source=points)
        plot = bokeh_server_renderer.get_plot(points)
        self.assertIsInstance(plot.callbacks[0], PointDrawCallback)
        callback = plot.callbacks[0]
        data = {'x': [1, 2, 3], 'y': [1, 2, 3], 'A': [None, None, 1]}
        callback.on_msg({'data': data})
        processed = dict(data, A=[np.nan, np.nan, 1])
        self.assertEqual(point_draw.element, Points(processed, vdims=['A']))

    def test_poly_draw_callback(self):
        polys = Polygons([[(0, 0), (2, 2), (4, 0)]])
        poly_draw = PolyDraw(source=polys)
        plot = bokeh_server_renderer.get_plot(polys)
        self.assertIsInstance(plot.callbacks[0], PolyDrawCallback)
        callback = plot.callbacks[0]
        data = {'x': [[1, 2, 3], [3, 4, 5]], 'y': [[1, 2, 3], [3, 4, 5]]}
        callback.on_msg({'data': data})
        element = Polygons([[(1, 1), (2, 2), (3, 3)], [(3, 3), (4, 4), (5, 5)]])
        self.assertEqual(poly_draw.element, element)

    def test_poly_draw_callback_initialized_server(self):
        polys = Polygons([[(0, 0), (2, 2), (4, 0)]])
        PolyDraw(source=polys)
        plot = bokeh_server_renderer.get_plot(polys)
        assert 'data' in plot.handles['source']._callbacks

    def test_poly_draw_callback_with_vdims(self):
        polys = Polygons([{'x': [0, 2, 4], 'y': [0, 2, 0], 'A': 1}], vdims=['A'])
        poly_draw = PolyDraw(source=polys)
        plot = bokeh_server_renderer.get_plot(polys)
        self.assertIsInstance(plot.callbacks[0], PolyDrawCallback)
        callback = plot.callbacks[0]
        data = {'x': [[1, 2, 3], [3, 4, 5]], 'y': [[1, 2, 3], [3, 4, 5]], 'A': [1, 2]}
        callback.on_msg({'data': data})
        element = Polygons([{'x': [1, 2, 3], 'y': [1, 2, 3], 'A': 1}, {'x': [3, 4, 5], 'y': [3, 4, 5], 'A': 2}], vdims=['A'])
        self.assertEqual(poly_draw.element, element)

    def test_poly_draw_callback_with_vdims_no_color_index(self):
        polys = Polygons([{'x': [0, 2, 4], 'y': [0, 2, 0], 'A': 1}], vdims=['A']).options(color_index=None)
        poly_draw = PolyDraw(source=polys)
        plot = bokeh_server_renderer.get_plot(polys)
        self.assertIsInstance(plot.callbacks[0], PolyDrawCallback)
        callback = plot.callbacks[0]
        data = {'x': [[1, 2, 3], [3, 4, 5]], 'y': [[1, 2, 3], [3, 4, 5]], 'A': [1, 2]}
        callback.on_msg({'data': data})
        element = Polygons([{'x': [1, 2, 3], 'y': [1, 2, 3], 'A': 1}, {'x': [3, 4, 5], 'y': [3, 4, 5], 'A': 2}], vdims=['A'])
        self.assertEqual(poly_draw.element, element)

    def test_box_edit_callback(self):
        boxes = Rectangles([(-0.5, -0.5, 0.5, 0.5)])
        box_edit = BoxEdit(source=boxes)
        plot = bokeh_server_renderer.get_plot(boxes)
        self.assertIsInstance(plot.callbacks[0], BoxEditCallback)
        callback = plot.callbacks[0]
        source = plot.handles['cds']
        self.assertEqual(source.data['left'], [-0.5])
        self.assertEqual(source.data['bottom'], [-0.5])
        self.assertEqual(source.data['right'], [0.5])
        self.assertEqual(source.data['top'], [0.5])
        data = {'left': [-0.25, 0], 'bottom': [-1, 0.75], 'right': [0.25, 2], 'top': [1, 1.25]}
        callback.on_msg({'data': data})
        element = Rectangles([(-0.25, -1, 0.25, 1), (0, 0.75, 2, 1.25)])
        self.assertEqual(box_edit.element, element)

    def test_box_edit_callback_legacy(self):
        boxes = Polygons([Box(0, 0, 1)])
        box_edit = BoxEdit(source=boxes)
        plot = bokeh_server_renderer.get_plot(boxes)
        self.assertIsInstance(plot.callbacks[0], BoxEditCallback)
        callback = plot.callbacks[0]
        source = plot.handles['cds']
        self.assertEqual(source.data['left'], [-0.5])
        self.assertEqual(source.data['bottom'], [-0.5])
        self.assertEqual(source.data['right'], [0.5])
        self.assertEqual(source.data['top'], [0.5])
        data = {'left': [-0.25, 0], 'bottom': [-1, 0.75], 'right': [0.25, 2], 'top': [1, 1.25]}
        callback.on_msg({'data': data})
        element = Polygons([Box(0, 0, (0.5, 2)), Box(1, 1, (2, 0.5))])
        self.assertEqual(box_edit.element, element)

    def test_box_edit_callback_initialized_server(self):
        boxes = Polygons([Box(0, 0, 1)])
        BoxEdit(source=boxes)
        plot = bokeh_server_renderer.get_plot(boxes)
        assert 'data' in plot.handles['cds']._callbacks

    @pytest.mark.flaky(reruns=3)
    def test_poly_edit_callback(self):
        polys = Polygons([[(0, 0), (2, 2), (4, 0)]])
        poly_edit = PolyEdit(source=polys)
        plot = bokeh_server_renderer.get_plot(polys)
        self.assertIsInstance(plot.callbacks[0], PolyEditCallback)
        callback = plot.callbacks[0]
        data = {'x': [[1, 2, 3], [3, 4, 5]], 'y': [[1, 2, 3], [3, 4, 5]]}
        callback.on_msg({'data': data})
        element = Polygons([[(1, 1), (2, 2), (3, 3)], [(3, 3), (4, 4), (5, 5)]])
        self.assertEqual(poly_edit.element, element)

    def test_poly_edit_callback_initialized_server(self):
        polys = Polygons([[(0, 0), (2, 2), (4, 0)]])
        PolyEdit(source=polys)
        plot = bokeh_server_renderer.get_plot(polys)
        assert 'data' in plot.handles['source']._callbacks

    def test_poly_edit_shared_callback(self):
        polys = Polygons([[(0, 0), (2, 2), (4, 0)]])
        polys2 = Polygons([[(0, 0), (2, 2), (4, 0)]])
        poly_edit = PolyEdit(source=polys, shared=True)
        poly_edit2 = PolyEdit(source=polys2, shared=True)
        plot = bokeh_server_renderer.get_plot(polys * polys2)
        edit_tools = [t for t in plot.state.tools if isinstance(t, PolyEditTool)]
        self.assertEqual(len(edit_tools), 1)
        plot1, plot2 = plot.subplots.values()
        self.assertIsInstance(plot1.callbacks[0], PolyEditCallback)
        callback = plot1.callbacks[0]
        data = {'x': [[1, 2, 3], [3, 4, 5]], 'y': [[1, 2, 3], [3, 4, 5]]}
        callback.on_msg({'data': data})
        self.assertIsInstance(plot2.callbacks[0], PolyEditCallback)
        callback = plot2.callbacks[0]
        data = {'x': [[1, 2, 3], [3, 4, 5]], 'y': [[1, 2, 3], [3, 4, 5]]}
        callback.on_msg({'data': data})
        element = Polygons([[(1, 1), (2, 2), (3, 3)], [(3, 3), (4, 4), (5, 5)]])
        self.assertEqual(poly_edit.element, element)
        self.assertEqual(poly_edit2.element, element)

    def test_point_draw_shared_datasource_callback(self):
        points = Points([1, 2, 3])
        table = Table(points.data, ['x', 'y'])
        layout = (points + table).opts(shared_datasource=True, clone=False)
        PointDraw(source=points)
        self.assertIs(points.data, table.data)
        plot = bokeh_renderer.get_plot(layout)
        point_plot = plot.subplots[0, 0].subplots['main']
        table_plot = plot.subplots[0, 1].subplots['main']
        self.assertIs(point_plot.handles['source'], table_plot.handles['source'])
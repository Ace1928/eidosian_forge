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
class TestServerCallbacks(CallbackTestCase):

    def test_server_callback_resolve_attr_spec_range1d_start(self):
        range1d = Range1d(start=0, end=10)
        msg = Callback.resolve_attr_spec('x_range.attributes.start', range1d)
        self.assertEqual(msg, {'id': range1d.ref['id'], 'value': 0})

    def test_server_callback_resolve_attr_spec_range1d_end(self):
        range1d = Range1d(start=0, end=10)
        msg = Callback.resolve_attr_spec('x_range.attributes.end', range1d)
        self.assertEqual(msg, {'id': range1d.ref['id'], 'value': 10})

    def test_server_callback_resolve_attr_spec_source_selected(self):
        source = ColumnDataSource()
        source.selected.indices = [1, 2, 3]
        msg = Callback.resolve_attr_spec('cb_obj.selected.indices', source)
        self.assertEqual(msg, {'id': source.ref['id'], 'value': [1, 2, 3]})

    def test_server_callback_resolve_attr_spec_tap_event(self):
        plot = Plot()
        event = Tap(plot, x=42)
        msg = Callback.resolve_attr_spec('cb_obj.x', event, plot)
        self.assertEqual(msg, {'id': plot.ref['id'], 'value': 42})

    def test_selection1d_resolves(self):
        points = Points([1, 2, 3])
        Selection1D(source=points)
        plot = bokeh_server_renderer.get_plot(points)
        selected = Selection(indices=[0, 2])
        callback = plot.callbacks[0]
        spec = callback.attributes['index']
        resolved = callback.resolve_attr_spec(spec, selected, model=selected)
        self.assertEqual(resolved, {'id': selected.ref['id'], 'value': [0, 2]})

    def test_selection1d_resolves_table(self):
        table = Table([1, 2, 3], 'x')
        Selection1D(source=table)
        plot = bokeh_server_renderer.get_plot(table)
        selected = Selection(indices=[0, 2])
        callback = plot.callbacks[0]
        spec = callback.attributes['index']
        resolved = callback.resolve_attr_spec(spec, selected, model=selected)
        self.assertEqual(resolved, {'id': selected.ref['id'], 'value': [0, 2]})

    def test_plotsize_resolves(self):
        points = Points([1, 2, 3])
        PlotSize(source=points)
        plot = bokeh_server_renderer.get_plot(points)
        callback = plot.callbacks[0]
        model = namedtuple('Plot', 'inner_width inner_height ref')(400, 300, {'id': 'Test'})
        width_spec = callback.attributes['width']
        height_spec = callback.attributes['height']
        resolved = callback.resolve_attr_spec(width_spec, model, model=model)
        self.assertEqual(resolved, {'id': 'Test', 'value': 400})
        resolved = callback.resolve_attr_spec(height_spec, model, model=model)
        self.assertEqual(resolved, {'id': 'Test', 'value': 300})

    def test_cds_resolves(self):
        points = Points([1, 2, 3])
        CDSStream(source=points)
        plot = bokeh_server_renderer.get_plot(points)
        cds = plot.handles['cds']
        callback = plot.callbacks[0]
        data_spec = callback.attributes['data']
        resolved = callback.resolve_attr_spec(data_spec, cds, model=cds)
        self.assertEqual(resolved, {'id': cds.ref['id'], 'value': points.columns()})

    def test_rangexy_datetime(self):
        df = pd.DataFrame(data=np.random.default_rng(2).standard_normal((30, 4)), columns=list('ABCD'), index=pd.date_range('2018-01-01', freq='D', periods=30))
        curve = Curve(df, 'index', 'C')
        stream = RangeXY(source=curve)
        plot = bokeh_server_renderer.get_plot(curve)
        callback = plot.callbacks[0]
        callback.on_msg({'x0': curve.iloc[0, 0], 'x1': curve.iloc[3, 0], 'y0': 0.2, 'y1': 0.8})
        self.assertEqual(stream.x_range[0], curve.iloc[0, 0])
        self.assertEqual(stream.x_range[1], curve.iloc[3, 0])
        self.assertEqual(stream.y_range, (0.2, 0.8))

    def test_rangexy_framewise_reset(self):
        raise SkipTest('The fix for this was reverted, see #4396')
        stream = RangeXY(x_range=(0, 2), y_range=(0, 1))
        curve = DynamicMap(lambda z, x_range, y_range: Curve([1, 2, z]), kdims=['z'], streams=[stream]).redim.range(z=(0, 3))
        plot = bokeh_server_renderer.get_plot(curve.opts(framewise=True))
        plot.update((1,))
        self.assertEqual(stream.y_range, None)

    def test_rangexy_framewise_not_reset_if_triggering(self):
        stream = RangeXY(x_range=(0, 2), y_range=(0, 1))
        curve = DynamicMap(lambda z, x_range, y_range: Curve([1, 2, z]), kdims=['z'], streams=[stream]).redim.range(z=(0, 3))
        bokeh_server_renderer.get_plot(curve.opts(framewise=True))
        stream.event(x_range=(0, 3))
        self.assertEqual(stream.x_range, (0, 3))
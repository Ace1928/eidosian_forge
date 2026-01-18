import unittest
import numpy as np
import holoviews as hv
from holoviews.element import (
from holoviews.plotting.bokeh.util import bokeh32, bokeh33
from .test_plot import TestBokehPlot, bokeh_renderer
class TestHVSpansPlot(TestBokehPlot):

    def setUp(self):
        if not bokeh32:
            raise unittest.SkipTest('Bokeh 3.2 added H/VSpans')
        super().setUp()

    def test_hspans_plot(self):
        hspans = HSpans({'y0': [0, 3, 5.5], 'y1': [1, 4, 6.5], 'extra': [-1, -2, -3]}, vdims=['extra'])
        plot = bokeh_renderer.get_plot(hspans)
        assert isinstance(plot.handles['glyph'], BkHStrip)
        assert plot.handles['xaxis'].axis_label == 'x'
        assert plot.handles['yaxis'].axis_label == 'y'
        assert plot.handles['x_range'].start == 0
        assert plot.handles['x_range'].end == 1
        assert plot.handles['y_range'].start == 0
        assert plot.handles['y_range'].end == 6.5
        source = plot.handles['source']
        assert list(source.data) == ['y0', 'y1']
        assert (source.data['y0'] == [0, 3, 5.5]).all()
        assert (source.data['y1'] == [1, 4, 6.5]).all()

    def test_hspans_plot_xlabel_ylabel(self):
        hspans = HSpans({'y0': [0, 3, 5.5], 'y1': [1, 4, 6.5], 'extra': [-1, -2, -3]}, vdims=['extra']).opts(xlabel='xlabel', ylabel='xlabel')
        plot = bokeh_renderer.get_plot(hspans)
        assert isinstance(plot.handles['glyph'], BkHStrip)
        assert plot.handles['xaxis'].axis_label == 'xlabel'
        assert plot.handles['yaxis'].axis_label == 'xlabel'

    def test_hspans_plot_invert_axes(self):
        hspans = HSpans({'y0': [0, 3, 5.5], 'y1': [1, 4, 6.5], 'extra': [-1, -2, -3]}, vdims=['extra']).opts(invert_axes=True)
        plot = bokeh_renderer.get_plot(hspans)
        assert isinstance(plot.handles['glyph'], BkVStrip)
        assert plot.handles['xaxis'].axis_label == 'y'
        assert plot.handles['yaxis'].axis_label == 'x'
        assert plot.handles['x_range'].start == 0
        assert plot.handles['x_range'].end == 6.5
        assert plot.handles['y_range'].start == 0
        assert plot.handles['y_range'].end == 1
        source = plot.handles['source']
        assert list(source.data) == ['y0', 'y1']
        assert (source.data['y0'] == [0, 3, 5.5]).all()
        assert (source.data['y1'] == [1, 4, 6.5]).all()

    def test_hspans_nondefault_kdims(self):
        hspans = HSpans({'other0': [0, 3, 5.5], 'other1': [1, 4, 6.5]}, kdims=['other0', 'other1'])
        plot = bokeh_renderer.get_plot(hspans)
        assert isinstance(plot.handles['glyph'], BkHStrip)
        assert plot.handles['xaxis'].axis_label == 'x'
        assert plot.handles['yaxis'].axis_label == 'y'
        assert plot.handles['x_range'].start == 0
        assert plot.handles['x_range'].end == 1
        assert plot.handles['y_range'].start == 0
        assert plot.handles['y_range'].end == 6.5
        source = plot.handles['source']
        assert list(source.data) == ['other0', 'other1']
        assert (source.data['other0'] == [0, 3, 5.5]).all()
        assert (source.data['other1'] == [1, 4, 6.5]).all()

    def test_vspans_plot(self):
        vspans = VSpans({'x0': [0, 3, 5.5], 'x1': [1, 4, 6.5], 'extra': [-1, -2, -3]}, vdims=['extra'])
        plot = bokeh_renderer.get_plot(vspans)
        assert isinstance(plot.handles['glyph'], BkVStrip)
        assert plot.handles['xaxis'].axis_label == 'x'
        assert plot.handles['yaxis'].axis_label == 'y'
        assert plot.handles['x_range'].start == 0
        assert plot.handles['x_range'].end == 6.5
        assert plot.handles['y_range'].start == 0
        assert plot.handles['y_range'].end == 1
        source = plot.handles['source']
        assert list(source.data) == ['x0', 'x1']
        assert (source.data['x0'] == [0, 3, 5.5]).all()
        assert (source.data['x1'] == [1, 4, 6.5]).all()

    def test_vspans_plot_invert_axes(self):
        vspans = VSpans({'x0': [0, 3, 5.5], 'x1': [1, 4, 6.5], 'extra': [-1, -2, -3]}, vdims=['extra']).opts(invert_axes=True)
        plot = bokeh_renderer.get_plot(vspans)
        assert isinstance(plot.handles['glyph'], BkHStrip)
        assert plot.handles['xaxis'].axis_label == 'y'
        assert plot.handles['yaxis'].axis_label == 'x'
        assert plot.handles['x_range'].start == 0
        assert plot.handles['x_range'].end == 1
        assert plot.handles['y_range'].start == 0
        assert plot.handles['y_range'].end == 6.5
        source = plot.handles['source']
        assert list(source.data) == ['x0', 'x1']
        assert (source.data['x0'] == [0, 3, 5.5]).all()
        assert (source.data['x1'] == [1, 4, 6.5]).all()

    def test_vspans_nondefault_kdims(self):
        vspans = VSpans({'other0': [0, 3, 5.5], 'other1': [1, 4, 6.5]}, kdims=['other0', 'other1'])
        plot = bokeh_renderer.get_plot(vspans)
        assert isinstance(plot.handles['glyph'], BkVStrip)
        assert plot.handles['xaxis'].axis_label == 'x'
        assert plot.handles['yaxis'].axis_label == 'y'
        assert plot.handles['x_range'].start == 0
        assert plot.handles['x_range'].end == 6.5
        assert plot.handles['y_range'].start == 0
        assert plot.handles['y_range'].end == 1
        source = plot.handles['source']
        assert list(source.data) == ['other0', 'other1']
        assert (source.data['other0'] == [0, 3, 5.5]).all()
        assert (source.data['other1'] == [1, 4, 6.5]).all()

    def test_dynamicmap_overlay_vspans(self):
        el = hv.VSpans(data=[[1, 3], [2, 4]])
        dmap = hv.DynamicMap(lambda: hv.Overlay([el]))
        plot_el = bokeh_renderer.get_plot(el)
        plot_dmap = bokeh_renderer.get_plot(dmap)
        assert plot_el.handles['x_range'].start == plot_dmap.handles['x_range'].start
        assert plot_el.handles['x_range'].end == plot_dmap.handles['x_range'].end
        assert plot_el.handles['y_range'].start == plot_dmap.handles['y_range'].start
        assert plot_el.handles['y_range'].end == plot_dmap.handles['y_range'].end

    def test_vspans_hspans_overlay(self):
        hspans = HSpans({'y0': [0, 3, 5.5], 'y1': [1, 4, 6.5], 'extra': [-1, -2, -3]}, vdims=['extra'])
        vspans = VSpans({'x0': [0, 3, 5.5], 'x1': [1, 4, 6.5], 'extra': [-1, -2, -3]}, vdims=['extra'])
        plot = bokeh_renderer.get_plot(hspans * vspans)
        assert plot.handles['xaxis'].axis_label == 'x'
        assert plot.handles['yaxis'].axis_label == 'y'
        assert plot.handles['x_range'].start == 0
        assert plot.handles['x_range'].end == 6.5
        assert plot.handles['y_range'].start == 0
        assert plot.handles['y_range'].end == 6.5

    def test_vlines_hlines_overlay_non_annotation(self):
        non_annotation = hv.Curve([], kdims=['time'])
        hspans = HSpans({'y0': [0, 3, 5.5], 'y1': [1, 4, 6.5], 'extra': [-1, -2, -3]}, vdims=['extra'])
        vspans = VSpans({'x0': [0, 3, 5.5], 'x1': [1, 4, 6.5], 'extra': [-1, -2, -3]}, vdims=['extra'])
        plot = bokeh_renderer.get_plot(non_annotation * hspans * vspans)
        assert plot.handles['xaxis'].axis_label == 'time'
        assert plot.handles['yaxis'].axis_label == 'y'

    def test_coloring_hline(self):
        hspans = HSpans({'y0': [1, 3, 5], 'y1': [2, 4, 6]}).opts(alpha=hv.dim('y0').norm(), line_color='red', line_dash=hv.dim('y1').bin([0, 3, 6], ['dashed', 'solid']))
        plot = hv.renderer('bokeh').get_plot(hspans)
        assert plot.handles['glyph'].line_color == 'red'
        data = plot.handles['glyph_renderer'].data_source.data
        np.testing.assert_allclose(data['alpha'], [0, 0.5, 1])
        assert data['line_dash'] == ['dashed', 'solid', 'solid']

    def test_dynamicmap_overlay_hspans(self):
        el = hv.HSpans(data=[[1, 3], [2, 4]])
        dmap = hv.DynamicMap(lambda: hv.Overlay([el]))
        plot_el = bokeh_renderer.get_plot(el)
        plot_dmap = bokeh_renderer.get_plot(dmap)
        assert plot_el.handles['x_range'].start == plot_dmap.handles['x_range'].start
        assert plot_el.handles['x_range'].end == plot_dmap.handles['x_range'].end
        assert plot_el.handles['y_range'].start == plot_dmap.handles['y_range'].start
        assert plot_el.handles['y_range'].end == plot_dmap.handles['y_range'].end
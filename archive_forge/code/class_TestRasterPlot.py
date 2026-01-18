from unittest import SkipTest
import numpy as np
import pandas as pd
import pytest
from bokeh.models import CustomJSHover, HoverTool
from holoviews.element import RGB, Image, ImageStack, Raster
from holoviews.plotting.bokeh.raster import ImageStackPlot
from holoviews.plotting.bokeh.util import bokeh34
from .test_plot import TestBokehPlot, bokeh_renderer
class TestRasterPlot(TestBokehPlot):

    def test_image_colormapping(self):
        img = Image(np.random.rand(10, 10)).opts(logz=True)
        self._test_colormapping(img, 2, True)

    def test_image_boolean_array(self):
        img = Image(np.array([[True, False], [False, True]]))
        plot = bokeh_renderer.get_plot(img)
        cmapper = plot.handles['color_mapper']
        source = plot.handles['source']
        assert cmapper.low == 0
        assert cmapper.high == 1
        np.testing.assert_equal(source.data['image'][0], np.array([[0, 1], [1, 0]]))

    def test_nodata_array(self):
        img = Image(np.array([[0, 1], [2, 0]])).opts(nodata=0)
        plot = bokeh_renderer.get_plot(img)
        cmapper = plot.handles['color_mapper']
        source = plot.handles['source']
        assert cmapper.low == 1
        assert cmapper.high == 2
        np.testing.assert_equal(source.data['image'][0], np.array([[2, np.nan], [np.nan, 1]]))

    def test_nodata_array_uint(self):
        img = Image(np.array([[0, 1], [2, 0]], dtype='uint32')).opts(nodata=0)
        plot = bokeh_renderer.get_plot(img)
        cmapper = plot.handles['color_mapper']
        source = plot.handles['source']
        assert cmapper.low == 1
        assert cmapper.high == 2
        np.testing.assert_equal(source.data['image'][0], np.array([[2, np.nan], [np.nan, 1]]))

    def test_nodata_rgb(self):
        N = 2
        rgb_d = np.linspace(0, 1, N * N * 3).reshape(N, N, 3)
        rgb = RGB(rgb_d).redim.nodata(R=0)
        plot = bokeh_renderer.get_plot(rgb)
        image_data = plot.handles['source'].data['image'][0]
        assert (image_data == 0).sum() == 1

    def test_raster_invert_axes(self):
        arr = np.array([[0, 1, 2], [3, 4, 5]])
        raster = Raster(arr).opts(invert_axes=True)
        plot = bokeh_renderer.get_plot(raster)
        source = plot.handles['source']
        np.testing.assert_equal(source.data['image'][0], arr.T)
        assert source.data['x'][0] == 0
        assert source.data['y'][0] == 0
        assert source.data['dw'][0] == 2
        assert source.data['dh'][0] == 3

    def test_image_invert_axes(self):
        arr = np.array([[0, 1, 2], [3, 4, 5]])
        raster = Image(arr).opts(invert_axes=True)
        plot = bokeh_renderer.get_plot(raster)
        source = plot.handles['source']
        np.testing.assert_equal(source.data['image'][0], np.rot90(arr)[::-1, ::-1])
        assert source.data['x'][0] == -0.5
        assert source.data['y'][0] == -0.5
        assert source.data['dw'][0] == 1
        assert source.data['dh'][0] == 1

    def test_image_invert_xaxis(self):
        arr = np.random.rand(10, 10)
        img = Image(arr).opts(invert_xaxis=True)
        plot = bokeh_renderer.get_plot(img)
        x_range = plot.handles['x_range']
        assert x_range.start == 0.5
        assert x_range.end == -0.5
        cdata = plot.handles['source'].data
        assert cdata['y'] == [-0.5]
        assert cdata['dh'] == [1.0]
        assert cdata['dw'] == [1.0]
        assert cdata['x'] == [-0.5]
        np.testing.assert_equal(cdata['image'][0], arr[::-1])

    def test_image_invert_yaxis(self):
        arr = np.random.rand(10, 10)
        img = Image(arr).opts(invert_yaxis=True)
        plot = bokeh_renderer.get_plot(img)
        y_range = plot.handles['y_range']
        assert y_range.start == 0.5
        assert y_range.end == -0.5
        cdata = plot.handles['source'].data
        assert cdata['x'] == [-0.5]
        assert cdata['dh'] == [1.0]
        assert cdata['dw'] == [1.0]
        assert cdata['y'] == [-0.5]
        np.testing.assert_equal(cdata['image'][0], arr[::-1])

    def test_rgb_invert_xaxis(self):
        rgb = RGB(np.random.rand(10, 10, 3)).opts(invert_xaxis=True)
        plot = bokeh_renderer.get_plot(rgb)
        x_range = plot.handles['x_range']
        assert x_range.start == 0.5
        assert x_range.end == -0.5
        cdata = plot.handles['source'].data
        assert cdata['y'] == [-0.5]
        assert cdata['dh'] == [1.0]
        assert cdata['dw'] == [1.0]
        assert cdata['x'] == [-0.5]

    def test_rgb_invert_yaxis(self):
        rgb = RGB(np.random.rand(10, 10, 3)).opts(invert_yaxis=True)
        plot = bokeh_renderer.get_plot(rgb)
        y_range = plot.handles['y_range']
        assert y_range.start == 0.5
        assert y_range.end == -0.5
        cdata = plot.handles['source'].data
        assert cdata['x'] == [-0.5]
        assert cdata['dh'] == [1.0]
        assert cdata['dw'] == [1.0]
        assert cdata['y'] == [-0.5]

    def test_image_datetime_hover(self):
        xr = pytest.importorskip('xarray')
        ts = pd.Timestamp('2020-01-01')
        data = xr.Dataset(coords={'x': [-0.5, 0.5], 'y': [-0.5, 0.5]}, data_vars={'Count': (['y', 'x'], [[0, 1], [2, 3]]), 'Timestamp': (['y', 'x'], [[ts, pd.NaT], [ts, ts]])})
        img = Image(data).opts(tools=['hover'])
        plot = bokeh_renderer.get_plot(img)
        hover = plot.handles['hover']
        assert hover.tooltips[-1] == ('Timestamp', '@{Timestamp}{%F %T}')
        assert '@{Timestamp}' in hover.formatters
        if bokeh34:
            assert hover.formatters['@{Timestamp}'] == 'datetime'
        else:
            assert isinstance(hover.formatters['@{Timestamp}'], CustomJSHover)

    def test_image_hover_with_custom_js(self):
        hover_tool = HoverTool(tooltips=[('x', '$x{custom}')], formatters={'x': CustomJSHover(code="return value + '2'")})
        img = Image(np.ones(100).reshape(10, 10)).opts(tools=[hover_tool])
        plot = bokeh_renderer.get_plot(img)
        hover = plot.handles['hover']
        assert hover.formatters == hover_tool.formatters
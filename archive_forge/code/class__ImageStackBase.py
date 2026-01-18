from unittest import SkipTest
import numpy as np
import pandas as pd
import pytest
from bokeh.models import CustomJSHover, HoverTool
from holoviews.element import RGB, Image, ImageStack, Raster
from holoviews.plotting.bokeh.raster import ImageStackPlot
from holoviews.plotting.bokeh.util import bokeh34
from .test_plot import TestBokehPlot, bokeh_renderer
class _ImageStackBase(TestRasterPlot):
    __test__ = False

    def test_image_stack_tuple(self):
        x = np.arange(self.xsize)
        y = np.arange(self.ysize) + 5
        a, b, c = (self.a, self.b, self.c)
        img_stack = ImageStack((x, y, a, b, c), kdims=['x', 'y'], vdims=['a', 'b', 'c'])
        plot = bokeh_renderer.get_plot(img_stack)
        source = plot.handles['source']
        np.testing.assert_equal(source.data['image'][0][:, :, 0], a)
        np.testing.assert_equal(source.data['image'][0][:, :, 1], b)
        np.testing.assert_equal(source.data['image'][0][:, :, 2], c)
        assert source.data['x'][0] == -0.5
        assert source.data['y'][0] == 4.5
        assert source.data['dw'][0] == self.xsize
        assert source.data['dh'][0] == self.ysize
        assert isinstance(plot, ImageStackPlot)

    def test_image_stack_tuple_unspecified_dims(self):
        x = np.arange(self.xsize)
        y = np.arange(self.ysize) + 5
        a, b, c = (self.a, self.b, self.c)
        img_stack = ImageStack((x, y, a, b, c), kdims=['x', 'y'])
        assert img_stack.vdims == ['level_0', 'level_1', 'level_2']
        plot = bokeh_renderer.get_plot(img_stack)
        source = plot.handles['source']
        np.testing.assert_equal(source.data['image'][0][:, :, 0], a)
        np.testing.assert_equal(source.data['image'][0][:, :, 1], b)
        np.testing.assert_equal(source.data['image'][0][:, :, 2], c)
        assert source.data['x'][0] == -0.5
        assert source.data['y'][0] == 4.5
        assert source.data['dw'][0] == self.xsize
        assert source.data['dh'][0] == self.ysize
        assert isinstance(plot, ImageStackPlot)

    def test_image_stack_dict(self):
        x = np.arange(self.xsize)
        y = np.arange(self.ysize) + 5
        a, b, c = (self.a, self.b, self.c)
        ds = {'x': x, 'y': y, 'a': a, 'b': b, 'c': c}
        img_stack = ImageStack(ds, kdims=['x', 'y'], vdims=['a', 'b', 'c'])
        plot = bokeh_renderer.get_plot(img_stack)
        source = plot.handles['source']
        np.testing.assert_equal(source.data['image'][0][:, :, 0], a)
        np.testing.assert_equal(source.data['image'][0][:, :, 1], b)
        np.testing.assert_equal(source.data['image'][0][:, :, 2], c)
        assert source.data['x'][0] == -0.5
        assert source.data['y'][0] == 4.5
        assert source.data['dw'][0] == self.xsize
        assert source.data['dh'][0] == self.ysize
        assert isinstance(plot, ImageStackPlot)

    def test_image_stack_dict_unspecified_dims(self):
        x = np.arange(self.xsize)
        y = np.arange(self.ysize) + 5
        a, b, c = (self.a, self.b, self.c)
        ds = {'x': x, 'y': y, 'a': a, 'b': b, 'c': c}
        img_stack = ImageStack(ds)
        plot = bokeh_renderer.get_plot(img_stack)
        source = plot.handles['source']
        np.testing.assert_equal(source.data['image'][0][:, :, 0], a)
        np.testing.assert_equal(source.data['image'][0][:, :, 1], b)
        np.testing.assert_equal(source.data['image'][0][:, :, 2], c)
        assert source.data['x'][0] == -0.5
        assert source.data['y'][0] == 4.5
        assert source.data['dw'][0] == self.xsize
        assert source.data['dh'][0] == self.ysize
        assert isinstance(plot, ImageStackPlot)

    def test_image_stack_dict_no_kdims(self):
        a, b, c = (self.a, self.b, self.c)
        ds = {'a': a, 'b': b, 'c': c}
        img_stack = ImageStack(ds)
        plot = bokeh_renderer.get_plot(img_stack)
        source = plot.handles['source']
        np.testing.assert_equal(source.data['image'][0][:, :, 0], a)
        np.testing.assert_equal(source.data['image'][0][:, :, 1], b)
        np.testing.assert_equal(source.data['image'][0][:, :, 2], c)
        assert source.data['x'][0] == -0.5
        assert source.data['y'][0] == -0.5
        assert source.data['dw'][0] == self.xsize
        assert source.data['dh'][0] == self.ysize
        assert isinstance(plot, ImageStackPlot)

    def test_image_stack_list(self):
        a, b, c = (self.a, self.b, self.c)
        ds = [a, b, c]
        img_stack = ImageStack(ds)
        plot = bokeh_renderer.get_plot(img_stack)
        source = plot.handles['source']
        np.testing.assert_equal(source.data['image'][0][:, :, 0], a)
        np.testing.assert_equal(source.data['image'][0][:, :, 1], b)
        np.testing.assert_equal(source.data['image'][0][:, :, 2], c)
        assert source.data['x'][0] == -0.5
        assert source.data['y'][0] == -0.5
        assert source.data['dw'][0] == self.xsize
        assert source.data['dh'][0] == self.ysize
        assert isinstance(plot, ImageStackPlot)

    def test_image_stack_xarray_dataset(self):
        try:
            import xarray as xr
        except ImportError:
            raise SkipTest('xarray not available for core tests')
        x = np.arange(self.xsize)
        y = np.arange(self.ysize) + 5
        a, b, c = (self.a, self.b, self.c)
        ds = xr.Dataset({'a': (['y', 'x'], a), 'b': (['y', 'x'], b), 'c': (['y', 'x'], c)}, coords={'x': x, 'y': y})
        img_stack = ImageStack(ds, kdims=['x', 'y'])
        plot = bokeh_renderer.get_plot(img_stack)
        source = plot.handles['source']
        np.testing.assert_equal(source.data['image'][0][:, :, 0], a)
        np.testing.assert_equal(source.data['image'][0][:, :, 1], b)
        np.testing.assert_equal(source.data['image'][0][:, :, 2], c)
        assert source.data['x'][0] == -0.5
        assert source.data['y'][0] == 4.5
        assert source.data['dw'][0] == self.xsize
        assert source.data['dh'][0] == self.ysize
        assert isinstance(plot, ImageStackPlot)

    def test_image_stack_xarray_dataarray(self):
        try:
            import xarray as xr
        except ImportError:
            raise SkipTest('xarray not available for core tests')
        x = np.arange(self.xsize)
        y = np.arange(self.ysize) + 5
        a, b, c = (self.a, self.b, self.c)
        da = xr.Dataset({'a': (['y', 'x'], a), 'b': (['y', 'x'], b), 'c': (['y', 'x'], c)}, coords={'x': x, 'y': y}).to_array('level')
        img_stack = ImageStack(da, vdims=['level'])
        plot = bokeh_renderer.get_plot(img_stack)
        source = plot.handles['source']
        np.testing.assert_equal(source.data['image'][0][:, :, 0], a)
        np.testing.assert_equal(source.data['image'][0][:, :, 1], b)
        np.testing.assert_equal(source.data['image'][0][:, :, 2], c)
        assert source.data['x'][0] == -0.5
        assert source.data['y'][0] == 4.5
        assert source.data['dw'][0] == self.xsize
        assert source.data['dh'][0] == self.ysize
        assert isinstance(plot, ImageStackPlot)

    def test_image_stack_invert_xaxis(self):
        x = np.arange(self.xsize)
        y = np.arange(self.ysize) + 5
        a, b, c = (self.a, self.b, self.c)
        img_stack = ImageStack((x, y, a, b, c), kdims=['x', 'y'], vdims=['a', 'b', 'c'])
        plot = bokeh_renderer.get_plot(img_stack.opts(invert_xaxis=True))
        source = plot.handles['source']
        np.testing.assert_equal(source.data['image'][0][:, :, 0], a)
        np.testing.assert_equal(source.data['image'][0][:, :, 1], b)
        np.testing.assert_equal(source.data['image'][0][:, :, 2], c)
        assert source.data['x'][0] == -0.5
        assert source.data['y'][0] == 4.5
        assert source.data['dw'][0] == self.xsize
        assert source.data['dh'][0] == self.ysize

    def test_image_stack_invert_yaxis(self):
        x = np.arange(self.xsize)
        y = np.arange(self.ysize) + 5
        a, b, c = (self.a, self.b, self.c)
        img_stack = ImageStack((x, y, a, b, c), kdims=['x', 'y'], vdims=['a', 'b', 'c'])
        plot = bokeh_renderer.get_plot(img_stack.opts(invert_yaxis=True))
        source = plot.handles['source']
        np.testing.assert_equal(source.data['image'][0][:, :, 0], a)
        np.testing.assert_equal(source.data['image'][0][:, :, 1], b)
        np.testing.assert_equal(source.data['image'][0][:, :, 2], c)
        assert source.data['x'][0] == -0.5
        assert source.data['y'][0] == 4.5
        assert source.data['dw'][0] == self.xsize
        assert source.data['dh'][0] == self.ysize

    def test_image_stack_invert_axes(self):
        x = np.arange(self.xsize)
        y = np.arange(self.ysize) + 5
        a, b, c = (self.a, self.b, self.c)
        img_stack = ImageStack((x, y, a, b, c), kdims=['x', 'y'], vdims=['a', 'b', 'c'])
        plot = bokeh_renderer.get_plot(img_stack.opts(invert_axes=True))
        source = plot.handles['source']
        np.testing.assert_equal(source.data['image'][0][:, :, 0].T, a)
        np.testing.assert_equal(source.data['image'][0][:, :, 1].T, b)
        np.testing.assert_equal(source.data['image'][0][:, :, 2].T, c)
        assert source.data['x'][0] == 4.5
        assert source.data['y'][0] == -0.5
        assert source.data['dw'][0] == self.ysize
        assert source.data['dh'][0] == self.xsize

    def test_image_stack_array(self):
        a, b, c = (self.a, self.b, self.c)
        data = np.dstack((a, b, c))
        img_stack = ImageStack(data, kdims=['x', 'y'], vdims=['a', 'b', 'c'])
        plot = bokeh_renderer.get_plot(img_stack)
        source = plot.handles['source']
        np.testing.assert_equal(source.data['image'][0][:, :, 0], a)
        np.testing.assert_equal(source.data['image'][0][:, :, 1], b)
        np.testing.assert_equal(source.data['image'][0][:, :, 2], c)
        assert source.data['x'][0] == -0.5
        assert source.data['y'][0] == -0.5
        assert source.data['dw'][0] == self.xsize
        assert source.data['dh'][0] == self.ysize
        assert isinstance(plot, ImageStackPlot)

    def test_image_stack_tuple_single_3darray(self):
        x = np.arange(self.xsize)
        y = np.arange(self.ysize) + 5
        a, b, c = (self.a, self.b, self.c)
        data = (x, y, np.dstack((a, b, c)))
        img_stack = ImageStack(data, kdims=['x', 'y'], vdims=['a', 'b', 'c'])
        plot = bokeh_renderer.get_plot(img_stack)
        source = plot.handles['source']
        np.testing.assert_equal(source.data['image'][0][:, :, 0], a)
        np.testing.assert_equal(source.data['image'][0][:, :, 1], b)
        np.testing.assert_equal(source.data['image'][0][:, :, 2], c)
        assert source.data['x'][0] == -0.5
        assert source.data['y'][0] == 4.5
        assert source.data['dw'][0] == self.xsize
        assert source.data['dh'][0] == self.ysize
        assert isinstance(plot, ImageStackPlot)
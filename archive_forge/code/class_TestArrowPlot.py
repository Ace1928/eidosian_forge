import unittest
import numpy as np
import holoviews as hv
from holoviews.element import (
from holoviews.plotting.bokeh.util import bokeh32, bokeh33
from .test_plot import TestBokehPlot, bokeh_renderer
class TestArrowPlot(TestBokehPlot):

    def _compare_arrow_plot(self, plot, start, end):
        print(plot.handles)
        arrow_glyph = plot.handles['arrow_1_glyph']
        arrow_cds = plot.handles['arrow_1_source']
        label_glyph = plot.handles['text_1_glyph']
        label_cds = plot.handles['text_1_source']
        x0, y0 = start
        x1, y1 = end
        self.assertEqual(label_glyph.x, 'x')
        self.assertEqual(label_glyph.y, 'y')
        self.assertEqual(label_cds.data, {'x': [x0], 'y': [y0], 'text': ['Test']})
        self.assertEqual(arrow_glyph.x_start, 'x_start')
        self.assertEqual(arrow_glyph.y_start, 'y_start')
        self.assertEqual(arrow_glyph.x_end, 'x_end')
        self.assertEqual(arrow_glyph.y_end, 'y_end')
        self.assertEqual(arrow_cds.data, {'x_start': [x0], 'x_end': [x1], 'y_start': [y0], 'y_end': [y1]})

    def test_arrow_plot_left(self):
        arrow = Arrow(0, 0, 'Test')
        plot = bokeh_renderer.get_plot(arrow)
        self._compare_arrow_plot(plot, (1 / 6.0, 0), (0, 0))

    def test_arrow_plot_up(self):
        arrow = Arrow(0, 0, 'Test', '^')
        plot = bokeh_renderer.get_plot(arrow)
        self._compare_arrow_plot(plot, (0, -1 / 6.0), (0, 0))

    def test_arrow_plot_right(self):
        arrow = Arrow(0, 0, 'Test', '>')
        plot = bokeh_renderer.get_plot(arrow)
        self._compare_arrow_plot(plot, (-1 / 6.0, 0), (0, 0))

    def test_arrow_plot_down(self):
        arrow = Arrow(0, 0, 'Test', 'v')
        plot = bokeh_renderer.get_plot(arrow)
        self._compare_arrow_plot(plot, (0, 1 / 6.0), (0, 0))
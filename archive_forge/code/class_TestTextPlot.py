import unittest
import numpy as np
import holoviews as hv
from holoviews.element import (
from holoviews.plotting.bokeh.util import bokeh32, bokeh33
from .test_plot import TestBokehPlot, bokeh_renderer
class TestTextPlot(TestBokehPlot):

    def test_text_plot(self):
        text = Text(0, 0, 'Test')
        plot = bokeh_renderer.get_plot(text)
        source = plot.handles['source']
        self.assertEqual(source.data, {'x': [0], 'y': [0], 'text': ['Test']})

    def test_text_plot_fontsize(self):
        text = Text(0, 0, 'Test', fontsize=18)
        plot = bokeh_renderer.get_plot(text)
        glyph = plot.handles['glyph']
        self.assertEqual(glyph.text_font_size, '18Pt')

    def test_text_plot_rotation(self):
        text = Text(0, 0, 'Test', rotation=90)
        plot = bokeh_renderer.get_plot(text)
        glyph = plot.handles['glyph']
        self.assertEqual(glyph.angle, np.pi / 2.0)

    def test_text_plot_rotation_style(self):
        text = Text(0, 0, 'Test').opts(angle=90)
        plot = bokeh_renderer.get_plot(text)
        glyph = plot.handles['glyph']
        self.assertEqual(glyph.angle, np.pi / 2.0)
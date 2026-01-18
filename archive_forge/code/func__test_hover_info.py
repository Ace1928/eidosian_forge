import numpy as np
import pyviz_comms as comms
from bokeh.models import (
from param import concrete_descendents
from holoviews import Curve
from holoviews.core.element import Element
from holoviews.core.options import Store
from holoviews.element.comparison import ComparisonTestCase
from holoviews.plotting.bokeh.callbacks import Callback
from holoviews.plotting.bokeh.element import ElementPlot
from .. import option_intersections
def _test_hover_info(self, element, tooltips, line_policy='nearest', formatters=None):
    if formatters is None:
        formatters = {}
    plot = bokeh_renderer.get_plot(element)
    plot.initialize_plot()
    fig = plot.state
    renderers = [r for r in plot.traverse(lambda x: x.handles.get('glyph_renderer')) if r is not None]
    hover = fig.select(dict(type=HoverTool))
    self.assertTrue(len(hover))
    self.assertEqual(hover[0].tooltips, tooltips)
    self.assertEqual(hover[0].formatters, formatters)
    self.assertEqual(hover[0].line_policy, line_policy)
    if isinstance(element, Element):
        cds = fig.select_one(dict(type=ColumnDataSource))
        for label, lookup in hover[0].tooltips:
            if label in element.dimensions():
                self.assertIn(lookup[2:-1], cds.data)
    print(renderers, hover)
    for renderer in renderers:
        self.assertTrue(any((renderer in h.renderers for h in hover)))
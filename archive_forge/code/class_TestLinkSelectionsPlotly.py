from unittest import skip, skipIf
import pandas as pd
import panel as pn
import holoviews as hv
from holoviews.core.options import Cycle, Store
from holoviews.element import ErrorBars, Points, Rectangles, Table, VSpan
from holoviews.element.comparison import ComparisonTestCase
from holoviews.plotting.util import linear_gradient
from holoviews.selection import link_selections
from holoviews.streams import SelectionXY
class TestLinkSelectionsPlotly(TestLinkSelections):
    __test__ = True

    def setUp(self):
        import holoviews.plotting.plotly
        super().setUp()
        self._backend = Store.current_backend
        Store.set_current_backend('plotly')

    def tearDown(self):
        Store.current_backend = self._backend

    def element_color(self, element, color_prop=None):
        if isinstance(element, Table):
            color = element.opts.get('style').kwargs['fill']
        elif isinstance(element, (Rectangles, VSpan)):
            color = element.opts.get('style').kwargs['line_color']
        else:
            color = element.opts.get('style').kwargs['color']
        if isinstance(color, (Cycle, str)):
            return color
        else:
            return list(color)
import asyncio
import time
import param
import pytest
from bokeh.client import pull_session
from bokeh.document import Document
from bokeh.io.doc import curdoc, set_curdoc
from bokeh.models import ColumnDataSource
from panel import serve
from panel.io.state import state
from panel.widgets import DiscreteSlider, FloatSlider
from holoviews.core.options import Store
from holoviews.core.spaces import DynamicMap
from holoviews.element import Curve, HLine, Path, Polygons
from holoviews.element.comparison import ComparisonTestCase
from holoviews.plotting import Renderer
from holoviews.plotting.bokeh.callbacks import Callback, RangeXYCallback, ResetCallback
from holoviews.plotting.bokeh.renderer import BokehRenderer
from holoviews.streams import PlotReset, RangeXY, Stream
def _launcher(self, obj, threaded=True, port=6001):
    try:
        asyncio.get_event_loop()
    except Exception:
        asyncio.set_event_loop(asyncio.new_event_loop())
    self._port = port
    server = serve(obj, threaded=threaded, show=False, port=port)
    time.sleep(0.5)
    return (server, self.session)
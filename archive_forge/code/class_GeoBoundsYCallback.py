import numpy as np
from pathlib import Path
from bokeh.models import CustomJS, CustomAction, PolyEditTool
from holoviews.core.ndmapping import UniformNdMapping
from holoviews.plotting.bokeh.callbacks import (
from holoviews.streams import (
from ...element.geo import _Element, Shape
from ...util import project_extents
from ...models import PolyVertexDrawTool, PolyVertexEditTool
from ...operation import project
from ...streams import PolyVertexEdit, PolyVertexDraw
from .plot import GeoOverlayPlot
class GeoBoundsYCallback(BoundsYCallback):

    def _process_msg(self, msg):
        msg = super()._process_msg(msg)
        if skip(self, msg, ('boundsy',)):
            return msg
        y0, y1 = msg['boundsy']
        plot = get_cb_plot(self)
        _, y0, _, y1 = project_extents((0, y0, 0, y1), plot.projection, plot.current_frame.crs)
        return {'boundsy': (y0, y1)}
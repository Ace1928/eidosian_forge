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
class GeoBoundsXYCallback(BoundsCallback):

    def _process_msg(self, msg):
        msg = super()._process_msg(msg)
        if skip(self, msg, ('bounds',)):
            return msg
        plot = get_cb_plot(self)
        msg['bounds'] = project_extents(msg['bounds'], plot.projection, plot.current_frame.crs)
        return msg
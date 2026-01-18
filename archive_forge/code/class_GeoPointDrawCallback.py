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
class GeoPointDrawCallback(PointDrawCallback):

    def _process_msg(self, msg):
        msg = super()._process_msg(msg)
        if not msg['data']:
            return msg
        projected = project_drawn(self, msg)
        if projected is None:
            return msg
        msg['data'] = projected.columns()
        return msg
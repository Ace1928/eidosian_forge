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
class GeoFreehandDrawCallback(FreehandDrawCallback):

    def _process_msg(self, msg):
        msg = super()._process_msg(msg)
        return project_poly(self, msg)

    def _update_cds_vdims(self, data):
        if isinstance(self.source, Shape):
            return
        super()._update_cds_vdims(data)
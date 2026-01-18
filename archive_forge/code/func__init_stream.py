import param
import cartopy.crs as ccrs
from holoviews.annotators import (
from holoviews.plotting.links import DataLink, VertexTableLink as hvVertexTableLink
from panel.util import param_name
from .element import Path
from .models.custom_tools import CheckpointTool, RestoreTool, ClearTool
from .links import VertexTableLink, PointTableLink, HvRectanglesTableLink, RectanglesTableLink
from .operation import project
from .streams import PolyVertexDraw, PolyVertexEdit
def _init_stream(self):
    name = param_name(self.name)
    style_kwargs = dict(node_style=self.node_style, feature_style=self.feature_style)
    self._stream = PolyVertexDraw(source=self.plot, data={}, num_objects=self.num_objects, show_vertices=self.show_vertices, tooltip='%s Tool' % name, **style_kwargs)
    if self.edit_vertices:
        self._vertex_stream = PolyVertexEdit(source=self.plot, tooltip='%s Edit Tool' % name, **style_kwargs)
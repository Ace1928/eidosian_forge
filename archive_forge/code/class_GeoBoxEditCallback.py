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
class GeoBoxEditCallback(BoxEditCallback):

    def _process_msg(self, msg):
        msg = super()._process_msg(msg)
        proj = self.plot.projection
        element = self.source
        if isinstance(element, UniformNdMapping):
            element = element.last
        if not isinstance(element, _Element) or element.crs == proj:
            return msg
        boxes = msg['data']
        data = dict(boxes, x0=[], y0=[], x1=[], y1=[])
        for extent in zip(boxes['x0'], boxes['y0'], boxes['x1'], boxes['y1']):
            x0, y0, x1, y1 = project_extents(extent, proj, element.crs)
            data['x0'].append(x0)
            data['y0'].append(y0)
            data['x1'].append(x1)
            data['y1'].append(y1)
        return {'data': data}
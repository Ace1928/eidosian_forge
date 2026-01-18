import asyncio
import base64
import time
from collections import defaultdict
import numpy as np
from bokeh.models import (
from panel.io.state import set_curdoc, state
from ...core.options import CallbackError
from ...core.util import datetime_types, dimension_sanitizer, dt64_to_dt, isequal
from ...element import Table
from ...streams import (
from ...util.warnings import warn
from .util import bokeh33, convert_timestamp
class PointDrawCallback(GlyphDrawCallback):

    def initialize(self, plot_id=None):
        plot = self.plot
        stream = self.streams[0]
        cds = plot.handles['source']
        glyph = plot.handles['glyph']
        renderers = [plot.handles['glyph_renderer']]
        kwargs = {}
        if stream.num_objects:
            kwargs['num_objects'] = stream.num_objects
        if stream.tooltip:
            kwargs['description'] = stream.tooltip
        if stream.styles:
            self._create_style_callback(cds, glyph)
        if stream.empty_value is not None:
            kwargs['empty_value'] = stream.empty_value
        point_tool = PointDrawTool(add=all((s.add for s in self.streams)), drag=all((s.drag for s in self.streams)), renderers=renderers, **kwargs)
        self.plot.state.tools.append(point_tool)
        self._update_cds_vdims(cds.data)
        super().initialize(plot_id)

    def _process_msg(self, msg):
        self._update_cds_vdims(msg['data'])
        return super()._process_msg(msg)
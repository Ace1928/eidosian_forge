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
class CurveEditCallback(GlyphDrawCallback):

    def initialize(self, plot_id=None):
        plot = self.plot
        stream = self.streams[0]
        cds = plot.handles['cds']
        glyph = plot.handles['glyph']
        renderer = plot.state.scatter(glyph.x, glyph.y, source=cds, visible=False, **stream.style)
        renderers = [renderer]
        kwargs = {}
        if stream.tooltip:
            kwargs['description'] = stream.tooltip
        point_tool = PointDrawTool(add=False, drag=True, renderers=renderers, **kwargs)
        code = 'renderer.visible = tool.active || (cds.selected.indices.length > 0)'
        show_vertices = CustomJS(args={'renderer': renderer, 'cds': cds, 'tool': point_tool}, code=code)
        point_tool.js_on_change('change:active', show_vertices)
        cds.selected.js_on_change('indices', show_vertices)
        self.plot.state.tools.append(point_tool)
        self._update_cds_vdims(cds.data)
        super().initialize(plot_id)

    def _process_msg(self, msg):
        self._update_cds_vdims(msg['data'])
        return super()._process_msg(msg)

    def _update_cds_vdims(self, data):
        """
        Add any value dimensions not already in the data ensuring the
        element can be reconstituted in entirety.
        """
        element = self.plot.current_frame
        for d in element.vdims:
            dim = dimension_sanitizer(d.name)
            if dim not in data:
                data[dim] = element.dimension_values(d)
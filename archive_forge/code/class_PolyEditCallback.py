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
class PolyEditCallback(PolyDrawCallback):

    def initialize(self, plot_id=None):
        plot = self.plot
        cds = plot.handles['cds']
        vertex_tool = None
        if all((s.shared for s in self.streams)):
            tools = [tool for tool in plot.state.tools if isinstance(tool, PolyEditTool)]
            vertex_tool = tools[0] if tools else None
        stream = self.streams[0]
        kwargs = {}
        if stream.tooltip:
            kwargs['description'] = stream.tooltip
        if vertex_tool is None:
            vertex_style = dict({'size': 10}, **stream.vertex_style)
            r1 = plot.state.scatter([], [], **vertex_style)
            vertex_tool = PolyEditTool(vertex_renderer=r1, **kwargs)
            plot.state.tools.append(vertex_tool)
        vertex_tool.renderers.append(plot.handles['glyph_renderer'])
        self._update_cds_vdims(cds.data)
        CDSCallback.initialize(self, plot_id)
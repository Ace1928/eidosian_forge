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
class PolyVertexEditCallback(GeoPolyEditCallback):
    split_code = '\n    var vcds = vertex.data_source\n    var vertices = vcds.selected.indices;\n    var pcds = poly.data_source;\n    var index = null;\n    for (let i = 0; i < pcds.data.xs.length; i++) {\n        if (pcds.data.xs[i] === vcds.data.x) {\n            index = i;\n        }\n    }\n    if ((index == null) || !vertices.length) {return}\n    var vertex = vertices[0];\n    for (const col of poly.data_source.columns()) {\n        var data = pcds.data[col][index];\n        var first = data.slice(0, vertex+1)\n        var second = data.slice(vertex)\n        pcds.data[col][index] = first\n        pcds.data[col].splice(index+1, 0, second)\n    }\n    for (const c of vcds.columns()) {\n      vcds.data[c] = [];\n    }\n    pcds.change.emit()\n    pcds.properties.data.change.emit()\n    pcds.selection_manager.clear();\n    vcds.change.emit()\n    vcds.properties.data.change.emit()\n    vcds.selection_manager.clear();\n    '
    icon = (Path(__file__).parents[2] / 'icons' / 'PolyBreak.png').resolve()

    def _create_vertex_split_link(self, action, poly_renderer, vertex_renderer, vertex_tool):
        cb = CustomJS(code=self.split_code, args={'poly': poly_renderer, 'vertex': vertex_renderer, 'tool': vertex_tool})
        action.callback = cb

    def initialize(self, plot_id=None):
        plot = self.plot
        stream = self.streams[0]
        element = self.plot.current_frame
        vertex_tool = None
        if all((s.shared for s in self.streams)):
            tools = [tool for tool in plot.state.tools if isinstance(tool, PolyEditTool)]
            vertex_tool = tools[0] if tools else None
        renderer = plot.handles['glyph_renderer']
        if vertex_tool is None:
            vertex_style = dict({'size': 10, 'alpha': 0.8}, **stream.vertex_style)
            r1 = plot.state.scatter([], [], **vertex_style)
            tooltip = '%s Edit Tool' % type(element).__name__
            vertex_tool = PolyVertexEditTool(vertex_renderer=r1, description=tooltip, node_style=stream.node_style, end_style=stream.feature_style)
            action = CustomAction(description='Split path', icon=self.icon)
            plot.state.add_tools(vertex_tool, action)
            self._create_vertex_split_link(action, renderer, r1, vertex_tool)
        vertex_tool.renderers.append(renderer)
        self._update_cds_vdims(renderer.data_source.data)
        CDSCallback.initialize(self, plot_id)
from collections import defaultdict
import numpy as np
import param
from bokeh.models import (
from ...core.data import Dataset
from ...core.options import Cycle, abbreviated_exception
from ...core.util import dimension_sanitizer, unique_array
from ...util.transform import dim
from ..mixins import ChordMixin, GraphMixin
from ..util import get_directed_graph_paths, process_cmap
from .chart import ColorbarPlot, PointPlot
from .element import CompositeElementPlot, LegendPlot
from .styles import (
class TriMeshPlot(GraphPlot):
    filled = param.Boolean(default=False, doc='\n        Whether the triangles should be drawn as filled.')
    style_opts = ['edge_' + p for p in base_properties + line_properties + fill_properties] + ['node_' + p for p in base_properties + fill_properties + line_properties] + ['node_size', 'cmap', 'edge_cmap', 'node_cmap']
    _node_columns = [0, 1, 2]

    def _process_vertices(self, element):
        style = self.style[self.cyclic_index]
        edge_color = style.get('edge_color')
        if edge_color not in element.nodes:
            edge_color = self.edge_color_index
        simplex_dim = element.get_dimension(edge_color)
        vertex_dim = element.nodes.get_dimension(edge_color)
        if vertex_dim and (not simplex_dim):
            simplices = element.array([0, 1, 2])
            z = element.nodes.dimension_values(vertex_dim)
            z = z[simplices].mean(axis=1)
            element = element.add_dimension(vertex_dim, len(element.vdims), z, vdim=True)
        element._initialize_edgepaths()
        return element

    def _init_glyphs(self, plot, element, ranges, source):
        element = self._process_vertices(element)
        super()._init_glyphs(plot, element, ranges, source)

    def _update_glyphs(self, element, ranges, style):
        element = self._process_vertices(element)
        super()._update_glyphs(element, ranges, style)
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
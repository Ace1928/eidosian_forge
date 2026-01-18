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
def _get_edge_paths(self, element, ranges):
    path_data, mapping = ({}, {})
    xidx, yidx = (1, 0) if self.invert_axes else (0, 1)
    if element._edgepaths is not None:
        edges = element._split_edgepaths.split(datatype='array', dimensions=element.edgepaths.kdims)
        if len(edges) == len(element):
            path_data['xs'] = [path[:, xidx] for path in edges]
            path_data['ys'] = [path[:, yidx] for path in edges]
            mapping = {'xs': 'xs', 'ys': 'ys'}
        else:
            raise ValueError('Edge paths do not match the number of supplied edges.Expected %d, found %d paths.' % (len(element), len(edges)))
    elif self.directed:
        xdim, ydim = element.nodes.kdims[:2]
        x_range = ranges[xdim.name]['combined']
        y_range = ranges[ydim.name]['combined']
        arrow_len = np.hypot(y_range[1] - y_range[0], x_range[1] - x_range[0]) * self.arrowhead_length
        arrows = get_directed_graph_paths(element, arrow_len)
        path_data['xs'] = [arr[:, 0] for arr in arrows]
        path_data['ys'] = [arr[:, 1] for arr in arrows]
    return (path_data, mapping)
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
def _get_edge_colors(self, element, ranges, edge_data, edge_mapping, style):
    cdim = element.get_dimension(self.edge_color_index)
    if not cdim:
        return
    elstyle = self.lookup_options(element, 'style')
    cycle = elstyle.kwargs.get('edge_color')
    if not isinstance(cycle, Cycle):
        cycle = None
    idx = element.get_dimension_index(cdim)
    field = dimension_sanitizer(cdim.name)
    cvals = element.dimension_values(cdim)
    if idx in self._node_columns:
        factors = element.nodes.dimension_values(2, expanded=False)
    elif idx == 2 and cvals.dtype.kind in 'uif':
        factors = None
    else:
        factors = unique_array(cvals)
    default_cmap = 'viridis' if factors is None else 'tab20'
    cmap = style.get('edge_cmap', style.get('cmap', default_cmap))
    nan_colors = {k: rgba_tuple(v) for k, v in self.clipping_colors.items()}
    if factors is None or (factors.dtype.kind in 'uif' and idx not in self._node_columns):
        colors, factors = (None, None)
    else:
        if factors.dtype.kind == 'f':
            cvals = cvals.astype(np.int32)
            factors = factors.astype(np.int32)
        if factors.dtype.kind not in 'SU':
            field += '_str__'
            cvals = [str(f) for f in cvals]
            factors = (str(f) for f in factors)
        factors = list(factors)
        if isinstance(cmap, dict):
            colors = [cmap.get(f, nan_colors.get('NaN', self._default_nan)) for f in factors]
        else:
            colors = process_cmap(cycle or cmap, len(factors))
    if field not in edge_data:
        edge_data[field] = cvals
    edge_style = dict(style, cmap=cmap)
    mapper = self._get_colormapper(cdim, element, ranges, edge_style, factors, colors, 'edge', 'edge_colormapper')
    transform = {'field': field, 'transform': mapper}
    color_type = 'fill_color' if self.filled else 'line_color'
    edge_mapping['edge_' + color_type] = transform
    edge_mapping['edge_nonselection_' + color_type] = transform
    edge_mapping['edge_selection_' + color_type] = transform
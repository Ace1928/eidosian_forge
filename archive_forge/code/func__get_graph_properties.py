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
def _get_graph_properties(self, plot, element, data, mapping, ranges, style):
    """Computes the args and kwargs for the GraphRenderer"""
    sources = []
    properties, mappings = ({}, {})
    for key in ('scatter_1', self.edge_glyph):
        gdata = data.pop(key, {})
        group_style = dict(style)
        style_group = self._style_groups.get('_'.join(key.split('_')[:-1]))
        with abbreviated_exception():
            group_style = self._apply_transforms(element, gdata, ranges, group_style, style_group)
        source = self._init_datasource(gdata)
        self.handles[key + '_source'] = source
        sources.append(source)
        others = [sg for sg in self._style_groups.values() if sg != style_group]
        glyph_props = self._glyph_properties(plot, element, source, ranges, group_style, style_group)
        for k, p in glyph_props.items():
            if any((k.startswith(o) for o in others)):
                continue
            properties[k] = p
        mappings.update(mapping.pop(key, {}))
    properties = {p: v for p, v in properties.items() if p != 'source' and 'legend' not in p}
    properties.update(mappings)
    layout = data.pop('layout', {})
    layout = StaticLayoutProvider(graph_layout=layout)
    self.handles['layout_source'] = layout
    return (tuple(sources + [layout]), properties)
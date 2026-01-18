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
def _sync_arcs(self):
    arc_renderer = self.handles['multi_line_2_glyph_renderer']
    scatter_renderer = self.handles['scatter_1_glyph_renderer']
    for gtype in ('selection_', 'nonselection_', 'muted_', 'hover_', ''):
        glyph = getattr(scatter_renderer, gtype + 'glyph')
        arc_glyph = getattr(arc_renderer, gtype + 'glyph')
        if not glyph or not arc_glyph:
            continue
        scatter_props = glyph.properties_with_values(include_defaults=False)
        styles = {k.replace('fill', 'line'): v for k, v in scatter_props.items() if 'fill' in k}
        arc_glyph.update(**styles)
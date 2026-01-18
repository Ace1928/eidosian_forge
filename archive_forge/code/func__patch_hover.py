import numpy as np
import param
from bokeh.models import Patches
from ...core.data import Dataset
from ...core.util import dimension_sanitizer, max_range
from ...util.transform import dim
from .graphs import GraphPlot
def _patch_hover(self, element, data):
    """
        Replace edge start and end hover data with label_index data.
        """
    if not (self.inspection_policy == 'edges' and 'hover' in self.handles):
        return
    lidx = element.nodes.get_dimension(self.label_index)
    src, tgt = (dimension_sanitizer(kd.name) for kd in element.kdims[:2])
    if src == 'start':
        src += '_values'
    if tgt == 'end':
        tgt += '_values'
    lookup = dict(zip(*(element.nodes.dimension_values(d) for d in (2, lidx))))
    src_vals = data['patches_1'][src]
    tgt_vals = data['patches_1'][tgt]
    data['patches_1'][src] = [lookup.get(v, v) for v in src_vals]
    data['patches_1'][tgt] = [lookup.get(v, v) for v in tgt_vals]
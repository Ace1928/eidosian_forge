from collections import defaultdict
import numpy as np
import param
from bokeh.models import CategoricalColorMapper, CustomJS, FactorRange, Range1d, Whisker
from bokeh.models.tools import BoxSelectTool
from bokeh.transform import jitter
from ...core.data import Dataset
from ...core.dimension import dimension_name
from ...core.util import dimension_sanitizer, isfinite
from ...operation import interpolate_curve
from ...util.transform import dim
from ..mixins import AreaMixin, BarsMixin, SpikesMixin
from ..util import compute_sizes, get_min_distance
from .element import ColorbarPlot, ElementPlot, LegendPlot, OverlayPlot
from .selection import BokehOverlaySelectionDisplay
from .styles import (
from .util import categorize_array
def _add_color_data(self, ds, ranges, style, cdim, data, mapping, factors, colors):
    cdata, cmapping = self._get_color_data(ds, ranges, dict(style), factors=factors, colors=colors)
    if 'color' not in cmapping:
        return
    cmapper = cmapping['color']['transform']
    legend_prop = 'legend_field'
    if 'color' in cmapping and self.show_legend and isinstance(cmapper, CategoricalColorMapper):
        mapping[legend_prop] = cdim.name
    if not self.stacked and ds.ndims > 1 and self.multi_level:
        cmapping.pop(legend_prop, None)
        mapping.pop(legend_prop, None)
    mapping.update(cmapping)
    for k, cd in cdata.items():
        if isinstance(cmapper, CategoricalColorMapper) and cd.dtype.kind in 'uif':
            cd = categorize_array(cd, cdim)
        if k not in data or len(data[k]) != next((len(data[key]) for key in data if key != k)):
            data[k].append(cd)
        else:
            data[k][-1] = cd
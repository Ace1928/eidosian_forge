import matplotlib as mpl
import numpy as np
import param
from matplotlib import cm
from matplotlib.collections import LineCollection
from matplotlib.dates import DateFormatter, date2num
from packaging.version import Version
from ...core.dimension import Dimension, dimension_name
from ...core.options import Store, abbreviated_exception
from ...core.util import (
from ...element import HeatMap, Raster
from ...operation import interpolate_curve
from ...util.transform import dim
from ..mixins import AreaMixin, BarsMixin, SpikesMixin
from ..plot import PlotSelector
from ..util import compute_sizes, get_min_distance, get_sideplot_ranges
from .element import ColorbarPlot, ElementPlot, LegendPlot
from .path import PathPlot
from .plot import AdjoinedPlot, mpl_rc_context
from .util import mpl_version
def _create_bars(self, axis, element, ranges, style):
    (gdim, cdim, sdim), values = self._get_values(element, ranges)
    style_dim = None
    if sdim:
        cats = values['stack']
        style_dim = sdim
    elif cdim:
        cats = values['category']
        style_dim = cdim
    if style_dim:
        style_map = {style_dim.pprint_value(v): self.style[i] for i, v in enumerate(cats)}
    else:
        style_map = {None: {}}
    width = (1 - 2.0 * self.bar_padding) / len(values.get('category', [None]))
    if self.invert_axes:
        plot_fn = 'barh'
        x, y, w, bottom = ('y', 'width', 'height', 'left')
    else:
        plot_fn = 'bar'
        x, y, w, bottom = ('x', 'height', 'width', 'bottom')
    xticks, labels, bar_data = ([], [], {})
    for gidx, grp in enumerate(values.get('group', [None])):
        sel_key = {}
        label = None
        if grp is not None:
            grp_label = gdim.pprint_value(grp)
            sel_key[gdim.name] = [grp]
            yalign = -0.04 if cdim and self.multi_level else 0
            xticks.append((gidx + 0.5, grp_label, yalign))
        for cidx, cat in enumerate(values.get('category', [None])):
            xpos = gidx + self.bar_padding + cidx * width
            if cat is not None:
                label = cdim.pprint_value(cat)
                sel_key[cdim.name] = [cat]
                if self.multi_level:
                    xticks.append((xpos + width / 2.0, label, 0))
            prev = 0
            for stk in values.get('stack', [None]):
                if stk is not None:
                    label = sdim.pprint_value(stk)
                    sel_key[sdim.name] = [stk]
                el = element.select(**sel_key)
                vals = el.dimension_values(element.vdims[0].name)
                val = float(vals[0]) if len(vals) else np.nan
                xval = xpos + width / 2.0
                if label in bar_data:
                    group = bar_data[label]
                    group[x].append(xval)
                    group[y].append(val)
                    group[bottom].append(prev)
                else:
                    bar_style = dict(style, **style_map.get(label, {}))
                    with abbreviated_exception():
                        bar_style = self._apply_transforms(el, ranges, bar_style)
                    bar_data[label] = {x: [xval], y: [val], w: width, bottom: [prev], 'label': label}
                    bar_data[label].update(bar_style)
                prev += val if isfinite(val) else 0
                if label is not None:
                    labels.append(label)
    bars = [getattr(axis, plot_fn)(**bar_spec) for bar_spec in bar_data.values()]
    ax_dims = [gdim]
    title = ''
    if sdim:
        title = sdim.pprint_label
        ax_dims.append(sdim)
    elif cdim:
        title = cdim.pprint_label
        if self.multi_level:
            ax_dims.append(cdim)
    if self.show_legend and any((len(l) for l in labels)) and (sdim or not self.multi_level):
        leg_spec = self.legend_specs[self.legend_position]
        if self.legend_cols:
            leg_spec['ncol'] = self.legend_cols
        legend_opts = self.legend_opts.copy()
        legend_opts.update(**leg_spec)
        axis.legend(title=title, **legend_opts)
    return (bars, xticks, ax_dims)
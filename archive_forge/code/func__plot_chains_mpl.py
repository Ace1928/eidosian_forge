import warnings
from itertools import cycle
from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
import matplotlib.ticker as mticker
from ....stats.density_utils import get_bins
from ...distplot import plot_dist
from ...plot_utils import _scale_fig_size, format_coords_as_labels
from ...rankplot import plot_rank
from . import backend_kwarg_defaults, backend_show, dealiase_sel_kwargs, matplotlib_kwarg_dealiaser
def _plot_chains_mpl(axes, idy, value, data, chain_prop, combined, xt_labelsize, rug, kind, trace_kwargs, hist_kwargs, plot_kwargs, fill_kwargs, rug_kwargs, rank_kwargs, circular, circ_var_units, circ_units_trace):
    if not circular:
        circ_var_units = False
    for chain_idx, row in enumerate(value):
        if kind == 'trace':
            aux_kwargs = dealiase_sel_kwargs(trace_kwargs, chain_prop, chain_idx)
            if idy:
                axes.plot(data.draw.values, row, **aux_kwargs)
                if circ_units_trace == 'degrees':
                    y_tick_locs = axes.get_yticks()
                    y_tick_labels = [i + 2 * 180 if i < 0 else i for i in np.rad2deg(y_tick_locs)]
                    axes.yaxis.set_major_locator(mticker.FixedLocator(y_tick_locs))
                    axes.set_yticklabels([f'{i:.0f}°' for i in y_tick_labels])
        if not combined:
            aux_kwargs = dealiase_sel_kwargs(plot_kwargs, chain_prop, chain_idx)
            if not idy:
                axes = plot_dist(values=row, textsize=xt_labelsize, rug=rug, ax=axes, hist_kwargs=hist_kwargs, plot_kwargs=aux_kwargs, fill_kwargs=fill_kwargs, rug_kwargs=rug_kwargs, backend='matplotlib', show=False, is_circular=circ_var_units)
    if kind == 'rank_bars' and idy:
        axes = plot_rank(data=value, kind='bars', ax=axes, **rank_kwargs)
    elif kind == 'rank_vlines' and idy:
        axes = plot_rank(data=value, kind='vlines', ax=axes, **rank_kwargs)
    if combined:
        aux_kwargs = dealiase_sel_kwargs(plot_kwargs, chain_prop, -1)
        if not idy:
            axes = plot_dist(values=value.flatten(), textsize=xt_labelsize, rug=rug, ax=axes, hist_kwargs=hist_kwargs, plot_kwargs=aux_kwargs, fill_kwargs=fill_kwargs, rug_kwargs=rug_kwargs, backend='matplotlib', show=False, is_circular=circ_var_units)
    return axes
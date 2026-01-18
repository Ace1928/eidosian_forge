import logging
import platform
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation, get_backend
from ....stats.density_utils import get_bins, histogram, kde
from ...kdeplot import plot_kde
from ...plot_utils import _scale_fig_size
from . import backend_kwarg_defaults, backend_show, create_axes_grid
def _set_animation(pp_sampled_vals, ax, dtype=None, kind='density', alpha=None, color=None, drawstyle=None, linewidth=None, height=None, markersize=None, plot_kwargs=None):
    if kind == 'kde':
        length = len(pp_sampled_vals)
        if dtype == 'f':
            x_vals, y_vals = kde(pp_sampled_vals[0])
            max_max = max((max(kde(pp_sampled_vals[i])[1]) for i in range(length)))
            ax.set_ylim(0, max_max)
            line, = ax.plot(x_vals, y_vals, **plot_kwargs)

            def animate(i):
                x_vals, y_vals = kde(pp_sampled_vals[i])
                line.set_data(x_vals, y_vals)
                return (line,)
        else:
            vals = pp_sampled_vals[0]
            bins = get_bins(vals)
            _, y_vals, x_vals = histogram(vals, bins=bins)
            line, = ax.plot(x_vals[:-1], y_vals, **plot_kwargs)
            max_max = max((max(histogram(pp_sampled_vals[i], bins=get_bins(pp_sampled_vals[i]))[1]) for i in range(length)))
            ax.set_ylim(0, max_max)

            def animate(i):
                pp_vals = pp_sampled_vals[i]
                _, y_vals, x_vals = histogram(pp_vals, bins=get_bins(pp_vals))
                line.set_data(x_vals[:-1], y_vals)
                return (line,)
    elif kind == 'cumulative':
        x_vals, y_vals = _empirical_cdf(pp_sampled_vals[0])
        line, = ax.plot(x_vals, y_vals, alpha=alpha, color=color, drawstyle=drawstyle, linewidth=linewidth)

        def animate(i):
            x_vals, y_vals = _empirical_cdf(pp_sampled_vals[i])
            line.set_data(x_vals, y_vals)
            return (line,)
    elif kind == 'scatter':
        x_vals = pp_sampled_vals[0]
        y_vals = np.full_like(x_vals, height, dtype=np.float64)
        line, = ax.plot(x_vals, y_vals, 'o', zorder=2, color=color, markersize=markersize, alpha=alpha)

        def animate(i):
            line.set_xdata(np.ravel(pp_sampled_vals[i]))
            return (line,)

    def init():
        if kind != 'scatter':
            line.set_data([], [])
        else:
            line.set_xdata([])
        return (line,)
    return (animate, init)
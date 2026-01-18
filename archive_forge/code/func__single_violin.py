import numpy as np
from scipy.stats import gaussian_kde
from . import utils
def _single_violin(ax, pos, pos_data, width, side, plot_opts):
    """"""
    bw_factor = plot_opts.get('bw_factor', None)

    def _violin_range(pos_data, plot_opts):
        """Return array with correct range, with which violins can be plotted."""
        cutoff = plot_opts.get('cutoff', False)
        cutoff_type = plot_opts.get('cutoff_type', 'std')
        cutoff_val = plot_opts.get('cutoff_val', 1.5)
        s = 0.0
        if not cutoff:
            if cutoff_type == 'std':
                s = cutoff_val * np.std(pos_data)
            else:
                s = cutoff_val
        x_lower = kde.dataset.min() - s
        x_upper = kde.dataset.max() + s
        return np.linspace(x_lower, x_upper, 100)
    pos_data = np.asarray(pos_data)
    kde = gaussian_kde(pos_data, bw_method=bw_factor)
    xvals = _violin_range(pos_data, plot_opts)
    violin = kde.evaluate(xvals)
    violin = width * violin / violin.max()
    if side == 'both':
        envelope_l, envelope_r = (-violin + pos, violin + pos)
    elif side == 'right':
        envelope_l, envelope_r = (pos, violin + pos)
    elif side == 'left':
        envelope_l, envelope_r = (-violin + pos, pos)
    else:
        msg = "`side` parameter should be one of {'left', 'right', 'both'}."
        raise ValueError(msg)
    ax.fill_betweenx(xvals, envelope_l, envelope_r, facecolor=plot_opts.get('violin_fc', '#66c2a5'), edgecolor=plot_opts.get('violin_ec', 'k'), lw=plot_opts.get('violin_lw', 1), alpha=plot_opts.get('violin_alpha', 0.5))
    return (xvals, violin)
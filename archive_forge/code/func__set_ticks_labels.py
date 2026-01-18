import numpy as np
from scipy.stats import gaussian_kde
from . import utils
def _set_ticks_labels(ax, data, labels, positions, plot_opts):
    """Set ticks and labels on horizontal axis."""
    ax.set_xlim([np.min(positions) - 0.5, np.max(positions) + 0.5])
    ax.set_xticks(positions)
    label_fontsize = plot_opts.get('label_fontsize')
    label_rotation = plot_opts.get('label_rotation')
    if label_fontsize or label_rotation:
        from matplotlib.artist import setp
    if labels is not None:
        if not len(labels) == len(data):
            msg = 'Length of `labels` should equal length of `data`.'
            raise ValueError(msg)
        xticknames = ax.set_xticklabels(labels)
        if label_fontsize:
            setp(xticknames, fontsize=label_fontsize)
        if label_rotation:
            setp(xticknames, rotation=label_rotation)
    return
from statsmodels.compat.python import lrange
import numpy as np
import statsmodels.tsa.vector_ar.util as util
def acorr_plot(acorr, linewidth=8, xlabel=None, ax=None):
    import matplotlib.pyplot as plt
    if ax is None:
        ax = plt.gca()
    if xlabel is None:
        xlabel = np.arange(len(acorr))
    ax.vlines(xlabel, [0], acorr, lw=linewidth)
    ax.axhline(0, color='k')
    ax.set_ylim([-1, 1])
    ax.set_xlim([-1, xlabel[-1] + 1])
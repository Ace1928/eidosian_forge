import warnings
from collections import namedtuple
import numpy as np
from scipy import optimize, stats
from scipy._lib._util import check_random_state
def _hist_plot(self, ax, fit_params):
    from matplotlib.ticker import MaxNLocator
    support = self._dist.support(*fit_params)
    lb = support[0] if np.isfinite(support[0]) else min(self._data)
    ub = support[1] if np.isfinite(support[1]) else max(self._data)
    pxf = 'PMF' if self.discrete else 'PDF'
    if self.discrete:
        x = np.arange(lb, ub + 2)
        y = self.pxf(x, *fit_params)
        ax.vlines(x[:-1], 0, y[:-1], label='Fitted Distribution PMF', color='C0')
        options = dict(density=True, bins=x, align='left', color='C1')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel('k')
        ax.set_ylabel('PMF')
    else:
        x = np.linspace(lb, ub, 200)
        y = self.pxf(x, *fit_params)
        ax.plot(x, y, '--', label='Fitted Distribution PDF', color='C0')
        options = dict(density=True, bins=50, align='mid', color='C1')
        ax.set_xlabel('x')
        ax.set_ylabel('PDF')
    if len(self._data) > 50 or self.discrete:
        ax.hist(self._data, label='Histogram of Data', **options)
    else:
        ax.plot(self._data, np.zeros_like(self._data), '*', label='Data', color='C1')
    ax.set_title(f'Fitted $\\tt {self._dist.name}$ {pxf} and Histogram')
    ax.legend(*ax.get_legend_handles_labels())
    return ax
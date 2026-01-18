import warnings
from collections import namedtuple
import numpy as np
from scipy import optimize, stats
from scipy._lib._util import check_random_state
def _cdf_plot(self, ax, fit_params):
    data = np.sort(self._data)
    ecdf = self._plotting_positions(len(self._data))
    ls = '--' if len(np.unique(data)) < 30 else '.'
    xlabel = 'k' if self.discrete else 'x'
    ax.step(data, ecdf, ls, label='Empirical CDF', color='C1', zorder=0)
    xlim = ax.get_xlim()
    q = np.linspace(*xlim, 300)
    tcdf = self._dist.cdf(q, *fit_params)
    ax.plot(q, tcdf, label='Fitted Distribution CDF', color='C0', zorder=1)
    ax.set_xlim(xlim)
    ax.set_ylim(0, 1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('CDF')
    ax.set_title(f'Fitted $\\tt {self._dist.name}$ and Empirical CDF')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1])
    return ax
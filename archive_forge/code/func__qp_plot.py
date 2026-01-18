import warnings
from collections import namedtuple
import numpy as np
from scipy import optimize, stats
from scipy._lib._util import check_random_state
def _qp_plot(self, ax, fit_params, qq):
    data = np.sort(self._data)
    ps = self._plotting_positions(len(self._data))
    if qq:
        qp = 'Quantiles'
        plot_type = 'Q-Q'
        x = self._dist.ppf(ps, *fit_params)
        y = data
    else:
        qp = 'Percentiles'
        plot_type = 'P-P'
        x = ps
        y = self._dist.cdf(data, *fit_params)
    ax.plot(x, y, '.', label=f'Fitted Distribution {plot_type}', color='C0', zorder=1)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    lim = [min(xlim[0], ylim[0]), max(xlim[1], ylim[1])]
    if not qq:
        lim = (max(lim[0], 0), min(lim[1], 1))
    if self.discrete and qq:
        q_min, q_max = (int(lim[0]), int(lim[1] + 1))
        q_ideal = np.arange(q_min, q_max)
        ax.plot(q_ideal, q_ideal, 'o', label='Reference', color='k', alpha=0.25, markerfacecolor='none', clip_on=True)
    elif self.discrete and (not qq):
        p_min, p_max = lim
        a, b = self._dist.support(*fit_params)
        p_min = max(p_min, 0 if np.isfinite(a) else 0.001)
        p_max = min(p_max, 1 if np.isfinite(b) else 1 - 0.001)
        q_min, q_max = self._dist.ppf([p_min, p_max], *fit_params)
        qs = np.arange(q_min - 1, q_max + 1)
        ps = self._dist.cdf(qs, *fit_params)
        ax.step(ps, ps, '-', label='Reference', color='k', alpha=0.25, clip_on=True)
    else:
        ax.plot(lim, lim, '-', label='Reference', color='k', alpha=0.25, clip_on=True)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel(f'Fitted $\\tt {self._dist.name}$ Theoretical {qp}')
    ax.set_ylabel(f'Data {qp}')
    ax.set_title(f'Fitted $\\tt {self._dist.name}$ {plot_type} Plot')
    ax.legend(*ax.get_legend_handles_labels())
    ax.set_aspect('equal')
    return ax
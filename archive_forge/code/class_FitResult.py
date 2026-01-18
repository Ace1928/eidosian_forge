import warnings
from collections import namedtuple
import numpy as np
from scipy import optimize, stats
from scipy._lib._util import check_random_state
class FitResult:
    """Result of fitting a discrete or continuous distribution to data

    Attributes
    ----------
    params : namedtuple
        A namedtuple containing the maximum likelihood estimates of the
        shape parameters, location, and (if applicable) scale of the
        distribution.
    success : bool or None
        Whether the optimizer considered the optimization to terminate
        successfully or not.
    message : str or None
        Any status message provided by the optimizer.

    """

    def __init__(self, dist, data, discrete, res):
        self._dist = dist
        self._data = data
        self.discrete = discrete
        self.pxf = getattr(dist, 'pmf', None) or getattr(dist, 'pdf', None)
        shape_names = [] if dist.shapes is None else dist.shapes.split(', ')
        if not discrete:
            FitParams = namedtuple('FitParams', shape_names + ['loc', 'scale'])
        else:
            FitParams = namedtuple('FitParams', shape_names + ['loc'])
        self.params = FitParams(*res.x)
        if res.success and (not np.isfinite(self.nllf())):
            res.success = False
            res.message = 'Optimization converged to parameter values that are inconsistent with the data.'
        self.success = getattr(res, 'success', None)
        self.message = getattr(res, 'message', None)

    def __repr__(self):
        keys = ['params', 'success', 'message']
        m = max(map(len, keys)) + 1
        return '\n'.join([key.rjust(m) + ': ' + repr(getattr(self, key)) for key in keys if getattr(self, key) is not None])

    def nllf(self, params=None, data=None):
        """Negative log-likelihood function

        Evaluates the negative of the log-likelihood function of the provided
        data at the provided parameters.

        Parameters
        ----------
        params : tuple, optional
            The shape parameters, location, and (if applicable) scale of the
            distribution as a single tuple. Default is the maximum likelihood
            estimates (``self.params``).
        data : array_like, optional
            The data for which the log-likelihood function is to be evaluated.
            Default is the data to which the distribution was fit.

        Returns
        -------
        nllf : float
            The negative of the log-likelihood function.

        """
        params = params if params is not None else self.params
        data = data if data is not None else self._data
        return self._dist.nnlf(theta=params, x=data)

    def plot(self, ax=None, *, plot_type='hist'):
        """Visually compare the data against the fitted distribution.

        Available only if `matplotlib` is installed.

        Parameters
        ----------
        ax : `matplotlib.axes.Axes`
            Axes object to draw the plot onto, otherwise uses the current Axes.
        plot_type : {"hist", "qq", "pp", "cdf"}
            Type of plot to draw. Options include:

            - "hist": Superposes the PDF/PMF of the fitted distribution
              over a normalized histogram of the data.
            - "qq": Scatter plot of theoretical quantiles against the
              empirical quantiles. Specifically, the x-coordinates are the
              values of the fitted distribution PPF evaluated at the
              percentiles ``(np.arange(1, n) - 0.5)/n``, where ``n`` is the
              number of data points, and the y-coordinates are the sorted
              data points.
            - "pp": Scatter plot of theoretical percentiles against the
              observed percentiles. Specifically, the x-coordinates are the
              percentiles ``(np.arange(1, n) - 0.5)/n``, where ``n`` is
              the number of data points, and the y-coordinates are the values
              of the fitted distribution CDF evaluated at the sorted
              data points.
            - "cdf": Superposes the CDF of the fitted distribution over the
              empirical CDF. Specifically, the x-coordinates of the empirical
              CDF are the sorted data points, and the y-coordinates are the
              percentiles ``(np.arange(1, n) - 0.5)/n``, where ``n`` is
              the number of data points.

        Returns
        -------
        ax : `matplotlib.axes.Axes`
            The matplotlib Axes object on which the plot was drawn.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy import stats
        >>> import matplotlib.pyplot as plt  # matplotlib must be installed
        >>> rng = np.random.default_rng()
        >>> data = stats.nbinom(5, 0.5).rvs(size=1000, random_state=rng)
        >>> bounds = [(0, 30), (0, 1)]
        >>> res = stats.fit(stats.nbinom, data, bounds)
        >>> ax = res.plot()  # save matplotlib Axes object

        The `matplotlib.axes.Axes` object can be used to customize the plot.
        See `matplotlib.axes.Axes` documentation for details.

        >>> ax.set_xlabel('number of trials')  # customize axis label
        >>> ax.get_children()[0].set_linewidth(5)  # customize line widths
        >>> ax.legend()
        >>> plt.show()
        """
        try:
            import matplotlib
        except ModuleNotFoundError as exc:
            message = 'matplotlib must be installed to use method `plot`.'
            raise ModuleNotFoundError(message) from exc
        plots = {'histogram': self._hist_plot, 'qq': self._qq_plot, 'pp': self._pp_plot, 'cdf': self._cdf_plot, 'hist': self._hist_plot}
        if plot_type.lower() not in plots:
            message = f'`plot_type` must be one of {set(plots.keys())}'
            raise ValueError(message)
        plot = plots[plot_type.lower()]
        if ax is None:
            import matplotlib.pyplot as plt
            ax = plt.gca()
        fit_params = np.atleast_1d(self.params)
        return plot(ax=ax, fit_params=fit_params)

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

    def _qq_plot(self, **kwargs):
        return self._qp_plot(qq=True, **kwargs)

    def _pp_plot(self, **kwargs):
        return self._qp_plot(qq=False, **kwargs)

    def _plotting_positions(self, n, a=0.5):
        k = np.arange(1, n + 1)
        return (k - a) / (n + 1 - 2 * a)

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
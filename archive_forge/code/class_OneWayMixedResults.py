import numpy as np
import numpy.linalg as L
from statsmodels.base.model import LikelihoodModelResults
from statsmodels.tools.decorators import cache_readonly
class OneWayMixedResults(LikelihoodModelResults):
    """Results class for OneWayMixed models

    """

    def __init__(self, model):
        self.model = model
        self.params = model.params

    @cache_readonly
    def llf(self):
        return self.model.logL(ML=True)

    @property
    def params_random_units(self):
        return self.model.params_random_units

    def cov_random(self):
        return self.model.cov_random()

    def mean_random(self, idx='lastexog'):
        if idx == 'lastexog':
            meanr = self.params[-self.model.k_exog_re:]
        elif isinstance(idx, list):
            if not len(idx) == self.model.k_exog_re:
                raise ValueError('length of idx different from k_exog_re')
            else:
                meanr = self.params[idx]
        else:
            meanr = np.zeros(self.model.k_exog_re)
        return meanr

    def std_random(self):
        return np.sqrt(np.diag(self.cov_random()))

    def plot_random_univariate(self, bins=None, use_loc=True):
        """create plot of marginal distribution of random effects

        Parameters
        ----------
        bins : int or bin edges
            option for bins in matplotlibs hist method. Current default is not
            very sophisticated. All distributions use the same setting for
            bins.
        use_loc : bool
            If True, then the distribution with mean given by the fixed
            effect is used.

        Returns
        -------
        Figure
            figure with subplots

        Notes
        -----
        What can make this fancier?

        Bin edges will not make sense if loc or scale differ across random
        effect distributions.

        """
        import matplotlib.pyplot as plt
        from scipy.stats import norm as normal
        fig = plt.figure()
        k = self.model.k_exog_re
        if k > 3:
            rows, cols = (int(np.ceil(k * 0.5)), 2)
        else:
            rows, cols = (k, 1)
        if bins is None:
            bins = 5 + 2 * self.model.n_units ** (1.0 / 3.0)
        if use_loc:
            loc = self.mean_random()
        else:
            loc = [0] * k
        scale = self.std_random()
        for ii in range(k):
            ax = fig.add_subplot(rows, cols, ii)
            freq, bins_, _ = ax.hist(loc[ii] + self.params_random_units[:, ii], bins=bins, normed=True)
            points = np.linspace(bins_[0], bins_[-1], 200)
            ax.set_title('Random Effect %d Marginal Distribution' % ii)
            ax.plot(points, normal.pdf(points, loc=loc[ii], scale=scale[ii]), 'r')
        return fig

    def plot_scatter_pairs(self, idx1, idx2, title=None, ax=None):
        """create scatter plot of two random effects

        Parameters
        ----------
        idx1, idx2 : int
            indices of the two random effects to display, corresponding to
            columns of exog_re
        title : None or string
            If None, then a default title is added
        ax : None or matplotlib axis instance
            If None, then a figure with one axis is created and returned.
            If ax is not None, then the scatter plot is created on it, and
            this axis instance is returned.

        Returns
        -------
        ax_or_fig : axis or figure instance
            see ax parameter

        Notes
        -----
        Still needs ellipse from estimated parameters

        """
        import matplotlib.pyplot as plt
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax_or_fig = fig
        re1 = self.params_random_units[:, idx1]
        re2 = self.params_random_units[:, idx2]
        ax.plot(re1, re2, 'o', alpha=0.75)
        if title is None:
            title = 'Random Effects %d and %d' % (idx1, idx2)
        ax.set_title(title)
        ax_or_fig = ax
        return ax_or_fig

    def plot_scatter_all_pairs(self, title=None):
        from statsmodels.graphics.plot_grids import scatter_ellipse
        if self.model.k_exog_re < 2:
            raise ValueError('less than two variables available')
        return scatter_ellipse(self.params_random_units, ell_kwds={'color': 'r'})
import numpy as np
class _UnivariateFunction:
    __doc__ = '%(description)s\n\n    Parameters\n    ----------\n    nobs : int\n        number of observations to simulate\n    x : None or 1d array\n        If x is given then it is used for the exogenous variable instead of\n        creating a random sample\n    distr_x : None or distribution instance\n        Only used if x is None. The rvs method is used to create a random\n        sample of the exogenous (explanatory) variable.\n    distr_noise : None or distribution instance\n        The rvs method is used to create a random sample of the errors.\n\n    Attributes\n    ----------\n    x : ndarray, 1-D\n        exogenous or explanatory variable. x is sorted.\n    y : ndarray, 1-D\n        endogenous or response variable\n    y_true : ndarray, 1-D\n        expected values of endogenous or response variable, i.e. values of y\n        without noise\n    func : callable\n        underlying function (defined by subclass)\n\n    %(ref)s\n    '

    def __init__(self, nobs=200, x=None, distr_x=None, distr_noise=None):
        if x is None:
            if distr_x is None:
                x = np.random.normal(loc=0, scale=self.s_x, size=nobs)
            else:
                x = distr_x.rvs(size=nobs)
            x.sort()
        self.x = x
        if distr_noise is None:
            noise = np.random.normal(loc=0, scale=self.s_noise, size=nobs)
        else:
            noise = distr_noise.rvs(size=nobs)
        if hasattr(self, 'het_scale'):
            noise *= self.het_scale(self.x)
        self.y_true = y_true = self.func(x)
        self.y = y_true + noise

    def plot(self, scatter=True, ax=None):
        """plot the mean function and optionally the scatter of the sample

        Parameters
        ----------
        scatter : bool
            If true, then add scatterpoints of sample to plot.
        ax : None or matplotlib axis instance
            If None, then a matplotlib.pyplot figure is created, otherwise
            the given axis, ax, is used.

        Returns
        -------
        Figure
            This is either the created figure instance or the one associated
            with ax if ax is given.

        """
        if ax is None:
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
        if scatter:
            ax.plot(self.x, self.y, 'o', alpha=0.5)
        xx = np.linspace(self.x.min(), self.x.max(), 100)
        ax.plot(xx, self.func(xx), lw=2, color='b', label='dgp mean')
        return ax.figure
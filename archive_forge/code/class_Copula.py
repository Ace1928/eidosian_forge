from abc import ABC, abstractmethod
import numpy as np
from scipy import stats
from statsmodels.graphics import utils
class Copula(ABC):
    """A generic Copula class meant for subclassing.

    Notes
    -----
    A function :math:`\\phi` on :math:`[0, \\infty]` is the Laplace-Stieltjes
    transform of a distribution function if and only if :math:`\\phi` is
    completely monotone and :math:`\\phi(0) = 1` [2]_.

    The following algorithm for sampling a ``d``-dimensional exchangeable
    Archimedean copula with generator :math:`\\phi` is due to Marshall, Olkin
    (1988) [1]_, where :math:`LS^{−1}(\\phi)` denotes the inverse
    Laplace-Stieltjes transform of :math:`\\phi`.

    From a mixture representation with respect to :math:`F`, the following
    algorithm may be derived for sampling Archimedean copulas, see [1]_.

    1. Sample :math:`V \\sim F = LS^{−1}(\\phi)`.
    2. Sample i.i.d. :math:`X_i \\sim U[0,1], i \\in \\{1,...,d\\}`.
    3. Return:math:`(U_1,..., U_d)`, where :math:`U_i = \\phi(−\\log(X_i)/V), i
       \\in \\{1, ...,d\\}`.

    Detailed properties of each copula can be found in [3]_.

    Instances of the class can access the attributes: ``rng`` for the random
    number generator (used for the ``seed``).

    **Subclassing**

    When subclassing `Copula` to create a new copula, ``__init__`` and
    ``random`` must be redefined.

    * ``__init__(theta)``: If the copula
      does not take advantage of a ``theta``, this parameter can be omitted.
    * ``random(n, random_state)``: draw ``n`` from the copula.
    * ``pdf(x)``: PDF from the copula.
    * ``cdf(x)``: CDF from the copula.

    References
    ----------
    .. [1] Marshall AW, Olkin I. “Families of Multivariate Distributions”,
      Journal of the American Statistical Association, 83, 834–841, 1988.
    .. [2] Marius Hofert. "Sampling Archimedean copulas",
      Universität Ulm, 2008.
    .. rvs[3] Harry Joe. "Dependence Modeling with Copulas", Monographs on
      Statistics and Applied Probability 134, 2015.

    """

    def __init__(self, k_dim=2):
        self.k_dim = k_dim

    def rvs(self, nobs=1, args=(), random_state=None):
        """Draw `n` in the half-open interval ``[0, 1)``.

        Marginals are uniformly distributed.

        Parameters
        ----------
        nobs : int, optional
            Number of samples to generate from the copula. Default is 1.
        args : tuple
            Arguments for copula parameters. The number of arguments depends
            on the copula.
        random_state : {None, int, numpy.random.Generator}, optional
            If `seed` is None then the legacy singleton NumPy generator.
            This will change after 0.13 to use a fresh NumPy ``Generator``,
            so you should explicitly pass a seeded ``Generator`` if you
            need reproducible results.
            If `seed` is an int, a new ``Generator`` instance is used,
            seeded with `seed`.
            If `seed` is already a ``Generator`` instance then that instance is
            used.

        Returns
        -------
        sample : array_like (nobs, d)
            Sample from the copula.

        See Also
        --------
        statsmodels.tools.rng_qrng.check_random_state
        """
        raise NotImplementedError

    @abstractmethod
    def pdf(self, u, args=()):
        """Probability density function of copula.

        Parameters
        ----------
        u : array_like, 2-D
            Points of random variables in unit hypercube at which method is
            evaluated.
            The second (or last) dimension should be the same as the dimension
            of the random variable, e.g. 2 for bivariate copula.
        args : tuple
            Arguments for copula parameters. The number of arguments depends
            on the copula.

        Returns
        -------
        pdf : ndarray, (nobs, k_dim)
            Copula pdf evaluated at points ``u``.
        """

    def logpdf(self, u, args=()):
        """Log of copula pdf, loglikelihood.

        Parameters
        ----------
        u : array_like, 2-D
            Points of random variables in unit hypercube at which method is
            evaluated.
            The second (or last) dimension should be the same as the dimension
            of the random variable, e.g. 2 for bivariate copula.
        args : tuple
            Arguments for copula parameters. The number of arguments depends
            on the copula.

        Returns
        -------
        cdf : ndarray, (nobs, k_dim)
            Copula log-pdf evaluated at points ``u``.
        """
        return np.log(self.pdf(u, *args))

    @abstractmethod
    def cdf(self, u, args=()):
        """Cumulative distribution function evaluated at points u.

        Parameters
        ----------
        u : array_like, 2-D
            Points of random variables in unit hypercube at which method is
            evaluated.
            The second (or last) dimension should be the same as the dimension
            of the random variable, e.g. 2 for bivariate copula.
        args : tuple
            Arguments for copula parameters. The number of arguments depends
            on the copula.

        Returns
        -------
        cdf : ndarray, (nobs, k_dim)
            Copula cdf evaluated at points ``u``.
        """

    def plot_scatter(self, sample=None, nobs=500, random_state=None, ax=None):
        """Sample the copula and plot.

        Parameters
        ----------
        sample : array-like, optional
            The sample to plot.  If not provided (the default), a sample
            is generated.
        nobs : int, optional
            Number of samples to generate from the copula.
        random_state : {None, int, numpy.random.Generator}, optional
            If `seed` is None then the legacy singleton NumPy generator.
            This will change after 0.13 to use a fresh NumPy ``Generator``,
            so you should explicitly pass a seeded ``Generator`` if you
            need reproducible results.
            If `seed` is an int, a new ``Generator`` instance is used,
            seeded with `seed`.
            If `seed` is already a ``Generator`` instance then that instance is
            used.
        ax : AxesSubplot, optional
            If given, this subplot is used to plot in instead of a new figure
            being created.

        Returns
        -------
        fig : Figure
            If `ax` is None, the created figure.  Otherwise the figure to which
            `ax` is connected.
        sample : array_like (n, d)
            Sample from the copula.

        See Also
        --------
        statsmodels.tools.rng_qrng.check_random_state
        """
        if self.k_dim != 2:
            raise ValueError('Can only plot 2-dimensional Copula.')
        if sample is None:
            sample = self.rvs(nobs=nobs, random_state=random_state)
        fig, ax = utils.create_mpl_ax(ax)
        ax.scatter(sample[:, 0], sample[:, 1])
        ax.set_xlabel('u')
        ax.set_ylabel('v')
        return (fig, sample)

    def plot_pdf(self, ticks_nbr=10, ax=None):
        """Plot the PDF.

        Parameters
        ----------
        ticks_nbr : int, optional
            Number of color isolines for the PDF. Default is 10.
        ax : AxesSubplot, optional
            If given, this subplot is used to plot in instead of a new figure
            being created.

        Returns
        -------
        fig : Figure
            If `ax` is None, the created figure.  Otherwise the figure to which
            `ax` is connected.

        """
        from matplotlib import pyplot as plt
        if self.k_dim != 2:
            import warnings
            warnings.warn('Plotting 2-dimensional Copula.')
        n_samples = 100
        eps = 0.0001
        uu, vv = np.meshgrid(np.linspace(eps, 1 - eps, n_samples), np.linspace(eps, 1 - eps, n_samples))
        points = np.vstack([uu.ravel(), vv.ravel()]).T
        data = self.pdf(points).T.reshape(uu.shape)
        min_ = np.nanpercentile(data, 5)
        max_ = np.nanpercentile(data, 95)
        fig, ax = utils.create_mpl_ax(ax)
        vticks = np.linspace(min_, max_, num=ticks_nbr)
        range_cbar = [min_, max_]
        cs = ax.contourf(uu, vv, data, vticks, antialiased=True, vmin=range_cbar[0], vmax=range_cbar[1])
        ax.set_xlabel('u')
        ax.set_ylabel('v')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        cbar = plt.colorbar(cs, ticks=vticks)
        cbar.set_label('p')
        fig.tight_layout()
        return fig

    def tau_simulated(self, nobs=1024, random_state=None):
        """Kendall's tau based on simulated samples.

        Returns
        -------
        tau : float
            Kendall's tau.

        """
        x = self.rvs(nobs, random_state=random_state)
        return stats.kendalltau(x[:, 0], x[:, 1])[0]

    def fit_corr_param(self, data):
        """Copula correlation parameter using Kendall's tau of sample data.

        Parameters
        ----------
        data : array_like
            Sample data used to fit `theta` using Kendall's tau.

        Returns
        -------
        corr_param : float
            Correlation parameter of the copula, ``theta`` in Archimedean and
            pearson correlation in elliptical.
            If k_dim > 2, then average tau is used.
        """
        x = np.asarray(data)
        if x.shape[1] == 2:
            tau = stats.kendalltau(x[:, 0], x[:, 1])[0]
        else:
            k = self.k_dim
            taus = [stats.kendalltau(x[..., i], x[..., j])[0] for i in range(k) for j in range(i + 1, k)]
            tau = np.mean(taus)
        return self._arg_from_tau(tau)

    def _arg_from_tau(self, tau):
        """Compute correlation parameter from tau.

        Parameters
        ----------
        tau : float
            Kendall's tau.

        Returns
        -------
        corr_param : float
            Correlation parameter of the copula, ``theta`` in Archimedean and
            pearson correlation in elliptical.

        """
        raise NotImplementedError
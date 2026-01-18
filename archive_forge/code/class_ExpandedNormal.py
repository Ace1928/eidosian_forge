import warnings
import numpy as np
from numpy.polynomial.hermite_e import HermiteE
from scipy.special import factorial
from scipy.stats import rv_continuous
import scipy.special as special
class ExpandedNormal(rv_continuous):
    """Construct the Edgeworth expansion pdf given cumulants.

    Parameters
    ----------
    cum : array_like
        `cum[j]` contains `(j+1)`-th cumulant: cum[0] is the mean,
        cum[1] is the variance and so on.

    Notes
    -----
    This is actually an asymptotic rather than convergent series, hence
    higher orders of the expansion may or may not improve the result.
    In a strongly non-Gaussian case, it is possible that the density
    becomes negative, especially far out in the tails.

    Examples
    --------
    Construct the 4th order expansion for the chi-square distribution using
    the known values of the cumulants:

    >>> import matplotlib.pyplot as plt
    >>> from scipy import stats
    >>> from scipy.special import factorial
    >>> df = 12
    >>> chi2_c = [2**(j-1) * factorial(j-1) * df for j in range(1, 5)]
    >>> edgw_chi2 = ExpandedNormal(chi2_c, name='edgw_chi2', momtype=0)

    Calculate several moments:
    >>> m, v = edgw_chi2.stats(moments='mv')
    >>> np.allclose([m, v], [df, 2 * df])
    True

    Plot the density function:
    >>> mu, sigma = df, np.sqrt(2*df)
    >>> x = np.linspace(mu - 3*sigma, mu + 3*sigma)
    >>> fig1 = plt.plot(x, stats.chi2.pdf(x, df=df), 'g-', lw=4, alpha=0.5)
    >>> fig2 = plt.plot(x, stats.norm.pdf(x, mu, sigma), 'b--', lw=4, alpha=0.5)
    >>> fig3 = plt.plot(x, edgw_chi2.pdf(x), 'r-', lw=2)
    >>> plt.show()

    References
    ----------
    .. [*] E.A. Cornish and R.A. Fisher, Moments and cumulants in the
         specification of distributions, Revue de l'Institut Internat.
         de Statistique. 5: 307 (1938), reprinted in
         R.A. Fisher, Contributions to Mathematical Statistics. Wiley, 1950.
    .. [*] https://en.wikipedia.org/wiki/Edgeworth_series
    .. [*] S. Blinnikov and R. Moessner, Expansions for nearly Gaussian
        distributions, Astron. Astrophys. Suppl. Ser. 130, 193 (1998)
    """

    def __init__(self, cum, name='Edgeworth expanded normal', **kwds):
        if len(cum) < 2:
            raise ValueError('At least two cumulants are needed.')
        self._coef, self._mu, self._sigma = self._compute_coefs_pdf(cum)
        self._herm_pdf = HermiteE(self._coef)
        if self._coef.size > 2:
            self._herm_cdf = HermiteE(-self._coef[1:])
        else:
            self._herm_cdf = lambda x: 0.0
        r = np.real_if_close(self._herm_pdf.roots())
        r = (r - self._mu) / self._sigma
        if r[(np.imag(r) == 0) & (np.abs(r) < 4)].any():
            mesg = 'PDF has zeros at %s ' % r
            warnings.warn(mesg, RuntimeWarning)
        kwds.update({'name': name, 'momtype': 0})
        super().__init__(**kwds)

    def _pdf(self, x):
        y = (x - self._mu) / self._sigma
        return self._herm_pdf(y) * _norm_pdf(y) / self._sigma

    def _cdf(self, x):
        y = (x - self._mu) / self._sigma
        return _norm_cdf(y) + self._herm_cdf(y) * _norm_pdf(y)

    def _sf(self, x):
        y = (x - self._mu) / self._sigma
        return _norm_sf(y) - self._herm_cdf(y) * _norm_pdf(y)

    def _compute_coefs_pdf(self, cum):
        mu, sigma = (cum[0], np.sqrt(cum[1]))
        lam = np.asarray(cum)
        for j, l in enumerate(lam):
            lam[j] /= cum[1] ** j
        coef = np.zeros(lam.size * 3 - 5)
        coef[0] = 1.0
        for s in range(lam.size - 2):
            for p in _faa_di_bruno_partitions(s + 1):
                term = sigma ** (s + 1)
                for m, k in p:
                    term *= np.power(lam[m + 1] / factorial(m + 2), k) / factorial(k)
                r = sum((k for m, k in p))
                coef[s + 1 + 2 * r] += term
        return (coef, mu, sigma)
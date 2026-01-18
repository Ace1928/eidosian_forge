import numpy as np
from scipy import optimize
from scipy.stats.mstats import mquantiles
from statsmodels.nonparametric.api import KDEMultivariate, KernelReg
from statsmodels.nonparametric._kernel_base import \
class TestFForm:
    """
    Nonparametric test for functional form.

    Parameters
    ----------
    endog : list
        Dependent variable (training set)
    exog : list of array_like objects
        The independent (right-hand-side) variables
    bw : array_like, str
        Bandwidths for exog or specify method for bandwidth selection
    fform : function
        The functional form ``y = g(b, x)`` to be tested. Takes as inputs
        the RHS variables `exog` and the coefficients ``b`` (betas)
        and returns a fitted ``y_hat``.
    var_type : str
        The type of the independent `exog` variables:

            - c: continuous
            - o: ordered
            - u: unordered

    estimator : function
        Must return the estimated coefficients b (betas). Takes as inputs
        ``(endog, exog)``.  E.g. least square estimator::

            lambda (x,y): np.dot(np.pinv(np.dot(x.T, x)), np.dot(x.T, y))

    References
    ----------
    See Racine, J.: "Consistent Significance Testing for Nonparametric
    Regression" Journal of Business & Economics Statistics.

    See chapter 12 in [1]  pp. 355-357.
    """

    def __init__(self, endog, exog, bw, var_type, fform, estimator, nboot=100):
        self.endog = endog
        self.exog = exog
        self.var_type = var_type
        self.fform = fform
        self.estimator = estimator
        self.nboot = nboot
        self.bw = KDEMultivariate(exog, bw=bw, var_type=var_type).bw
        self.sig = self._compute_sig()

    def _compute_sig(self):
        Y = self.endog
        X = self.exog
        b = self.estimator(Y, X)
        m = self.fform(X, b)
        n = np.shape(X)[0]
        resid = Y - m
        resid = resid - np.mean(resid)
        self.test_stat = self._compute_test_stat(resid)
        sqrt5 = np.sqrt(5.0)
        fct1 = (1 - sqrt5) / 2.0
        fct2 = (1 + sqrt5) / 2.0
        u1 = fct1 * resid
        u2 = fct2 * resid
        r = fct2 / sqrt5
        I_dist = np.empty((self.nboot, 1))
        for j in range(self.nboot):
            u_boot = u2.copy()
            prob = np.random.uniform(0, 1, size=(n,))
            ind = prob < r
            u_boot[ind] = u1[ind]
            Y_boot = m + u_boot
            b_hat = self.estimator(Y_boot, X)
            m_hat = self.fform(X, b_hat)
            u_boot_hat = Y_boot - m_hat
            I_dist[j] = self._compute_test_stat(u_boot_hat)
        self.boots_results = I_dist
        sig = 'Not Significant'
        if self.test_stat > mquantiles(I_dist, 0.9):
            sig = '*'
        if self.test_stat > mquantiles(I_dist, 0.95):
            sig = '**'
        if self.test_stat > mquantiles(I_dist, 0.99):
            sig = '***'
        return sig

    def _compute_test_stat(self, u):
        n = np.shape(u)[0]
        XLOO = LeaveOneOut(self.exog)
        uLOO = LeaveOneOut(u[:, None]).__iter__()
        ival = 0
        S2 = 0
        for i, X_not_i in enumerate(XLOO):
            u_j = next(uLOO)
            u_j = np.squeeze(u_j)
            K = gpke(self.bw, data=-X_not_i, data_predict=-self.exog[i, :], var_type=self.var_type, tosum=False)
            f_i = u[i] * u_j * K
            assert u_j.shape == K.shape
            ival += f_i.sum()
            S2 += (f_i ** 2).sum()
            assert np.size(ival) == 1
            assert np.size(S2) == 1
        ival *= 1.0 / (n * (n - 1))
        ix_cont = _get_type_pos(self.var_type)[0]
        hp = self.bw[ix_cont].prod()
        S2 *= 2 * hp / (n * (n - 1))
        T = n * ival * np.sqrt(hp / S2)
        return T
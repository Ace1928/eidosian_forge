import copy
import numpy as np
from scipy import optimize
from scipy.stats.mstats import mquantiles
from ._kernel_base import GenericKDE, EstimatorSettings, gpke, \
class KernelCensoredReg(KernelReg):
    """
    Nonparametric censored regression.

    Calculates the conditional mean ``E[y|X]`` where ``y = g(X) + e``,
    where y is left-censored.  Left censored variable Y is defined as
    ``Y = min {Y', L}`` where ``L`` is the value at which ``Y`` is censored
    and ``Y'`` is the true value of the variable.

    Parameters
    ----------
    endog : list with one element which is array_like
        This is the dependent variable.
    exog : list
        The training data for the independent variable(s)
        Each element in the list is a separate variable
    dep_type : str
        The type of the dependent variable(s)
        c: Continuous
        u: Unordered (Discrete)
        o: Ordered (Discrete)
    reg_type : str
        Type of regression estimator
        lc: Local Constant Estimator
        ll: Local Linear Estimator
    bw : array_like
        Either a user-specified bandwidth or
        the method for bandwidth selection.
        cv_ls: cross-validation least squares
        aic: AIC Hurvich Estimator
    ckertype : str, optional
        The kernel used for the continuous variables.
    okertype : str, optional
        The kernel used for the ordered discrete variables.
    ukertype : str, optional
        The kernel used for the unordered discrete variables.
    censor_val : float
        Value at which the dependent variable is censored
    defaults : EstimatorSettings instance, optional
        The default values for the efficient bandwidth estimation

    Attributes
    ----------
    bw : array_like
        The bandwidth parameters
    """

    def __init__(self, endog, exog, var_type, reg_type, bw='cv_ls', ckertype='gaussian', ukertype='aitchison_aitken_reg', okertype='wangryzin_reg', censor_val=0, defaults=None):
        self.var_type = var_type
        self.data_type = var_type
        self.reg_type = reg_type
        self.ckertype = ckertype
        self.okertype = okertype
        self.ukertype = ukertype
        if not (self.ckertype in kernel_func and self.ukertype in kernel_func and (self.okertype in kernel_func)):
            raise ValueError('user specified kernel must be a supported kernel from statsmodels.nonparametric.kernels.')
        self.k_vars = len(self.var_type)
        self.endog = _adjust_shape(endog, 1)
        self.exog = _adjust_shape(exog, self.k_vars)
        self.data = np.column_stack((self.endog, self.exog))
        self.nobs = np.shape(self.exog)[0]
        self.est = dict(lc=self._est_loc_constant, ll=self._est_loc_linear)
        defaults = EstimatorSettings() if defaults is None else defaults
        self._set_defaults(defaults)
        self.censor_val = censor_val
        if self.censor_val is not None:
            self.censored(censor_val)
        else:
            self.W_in = np.ones((self.nobs, 1))
        if not self.efficient:
            self.bw = self._compute_reg_bw(bw)
        else:
            self.bw = self._compute_efficient(bw)

    def censored(self, censor_val):
        self.d = (self.endog != censor_val) * 1.0
        ix = np.argsort(np.squeeze(self.endog))
        self.sortix = ix
        self.sortix_rev = np.zeros(ix.shape, int)
        self.sortix_rev[ix] = np.arange(len(ix))
        self.endog = np.squeeze(self.endog[ix])
        self.endog = _adjust_shape(self.endog, 1)
        self.exog = np.squeeze(self.exog[ix])
        self.d = np.squeeze(self.d[ix])
        self.W_in = np.empty((self.nobs, 1))
        for i in range(1, self.nobs + 1):
            P = 1
            for j in range(1, i):
                P *= ((self.nobs - j) / (float(self.nobs) - j + 1)) ** self.d[j - 1]
            self.W_in[i - 1, 0] = P * self.d[i - 1] / (float(self.nobs) - i + 1)

    def __repr__(self):
        """Provide something sane to print."""
        rpr = 'KernelCensoredReg instance\n'
        rpr += 'Number of variables: k_vars = ' + str(self.k_vars) + '\n'
        rpr += 'Number of samples:   nobs = ' + str(self.nobs) + '\n'
        rpr += 'Variable types:      ' + self.var_type + '\n'
        rpr += 'BW selection method: ' + self._bw_method + '\n'
        rpr += 'Estimator type: ' + self.reg_type + '\n'
        return rpr

    def _est_loc_linear(self, bw, endog, exog, data_predict, W):
        """
        Local linear estimator of g(x) in the regression ``y = g(x) + e``.

        Parameters
        ----------
        bw : array_like
            Vector of bandwidth value(s)
        endog : 1D array_like
            The dependent variable
        exog : 1D or 2D array_like
            The independent variable(s)
        data_predict : 1D array_like of length K, where K is
            the number of variables. The point at which
            the density is estimated

        Returns
        -------
        D_x : array_like
            The value of the conditional mean at data_predict

        Notes
        -----
        See p. 81 in [1] and p.38 in [2] for the formulas
        Unlike other methods, this one requires that data_predict be 1D
        """
        nobs, k_vars = exog.shape
        ker = gpke(bw, data=exog, data_predict=data_predict, var_type=self.var_type, ckertype=self.ckertype, ukertype=self.ukertype, okertype=self.okertype, tosum=False)
        ker = W * ker[:, np.newaxis]
        M12 = exog - data_predict
        M22 = np.dot(M12.T, M12 * ker)
        M12 = (M12 * ker).sum(axis=0)
        M = np.empty((k_vars + 1, k_vars + 1))
        M[0, 0] = ker.sum()
        M[0, 1:] = M12
        M[1:, 0] = M12
        M[1:, 1:] = M22
        ker_endog = ker * endog
        V = np.empty((k_vars + 1, 1))
        V[0, 0] = ker_endog.sum()
        V[1:, 0] = ((exog - data_predict) * ker_endog).sum(axis=0)
        mean_mfx = np.dot(np.linalg.pinv(M), V)
        mean = mean_mfx[0]
        mfx = mean_mfx[1:, :]
        return (mean, mfx)

    def cv_loo(self, bw, func):
        """
        The cross-validation function with leave-one-out
        estimator

        Parameters
        ----------
        bw : array_like
            Vector of bandwidth values
        func : callable function
            Returns the estimator of g(x).
            Can be either ``_est_loc_constant`` (local constant) or
            ``_est_loc_linear`` (local_linear).

        Returns
        -------
        L : float
            The value of the CV function

        Notes
        -----
        Calculates the cross-validation least-squares
        function. This function is minimized by compute_bw
        to calculate the optimal value of bw

        For details see p.35 in [2]

        .. math:: CV(h)=n^{-1}\\sum_{i=1}^{n}(Y_{i}-g_{-i}(X_{i}))^{2}

        where :math:`g_{-i}(X_{i})` is the leave-one-out estimator of g(X)
        and :math:`h` is the vector of bandwidths
        """
        LOO_X = LeaveOneOut(self.exog)
        LOO_Y = LeaveOneOut(self.endog).__iter__()
        LOO_W = LeaveOneOut(self.W_in).__iter__()
        L = 0
        for ii, X_not_i in enumerate(LOO_X):
            Y = next(LOO_Y)
            w = next(LOO_W)
            G = func(bw, endog=Y, exog=-X_not_i, data_predict=-self.exog[ii, :], W=w)[0]
            L += (self.endog[ii] - G) ** 2
        return L / self.nobs

    def fit(self, data_predict=None):
        """
        Returns the marginal effects at the data_predict points.
        """
        func = self.est[self.reg_type]
        if data_predict is None:
            data_predict = self.exog
        else:
            data_predict = _adjust_shape(data_predict, self.k_vars)
        N_data_predict = np.shape(data_predict)[0]
        mean = np.empty((N_data_predict,))
        mfx = np.empty((N_data_predict, self.k_vars))
        for i in range(N_data_predict):
            mean_mfx = func(self.bw, self.endog, self.exog, data_predict=data_predict[i, :], W=self.W_in)
            mean[i] = np.squeeze(mean_mfx[0])
            mfx_c = np.squeeze(mean_mfx[1])
            mfx[i, :] = mfx_c
        return (mean, mfx)
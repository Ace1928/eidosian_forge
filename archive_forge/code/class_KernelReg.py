import copy
import numpy as np
from scipy import optimize
from scipy.stats.mstats import mquantiles
from ._kernel_base import GenericKDE, EstimatorSettings, gpke, \
class KernelReg(GenericKDE):
    """
    Nonparametric kernel regression class.

    Calculates the conditional mean ``E[y|X]`` where ``y = g(X) + e``.
    Note that the "local constant" type of regression provided here is also
    known as Nadaraya-Watson kernel regression; "local linear" is an extension
    of that which suffers less from bias issues at the edge of the support. Note
    that specifying a custom kernel works only with "local linear" kernel
    regression. For example, a custom ``tricube`` kernel yields LOESS regression.

    Parameters
    ----------
    endog : array_like
        This is the dependent variable.
    exog : array_like
        The training data for the independent variable(s)
        Each element in the list is a separate variable
    var_type : str
        The type of the variables, one character per variable:

            - c: continuous
            - u: unordered (discrete)
            - o: ordered (discrete)

    reg_type : {'lc', 'll'}, optional
        Type of regression estimator. 'lc' means local constant and
        'll' local Linear estimator.  Default is 'll'
    bw : str or array_like, optional
        Either a user-specified bandwidth or the method for bandwidth
        selection. If a string, valid values are 'cv_ls' (least-squares
        cross-validation) and 'aic' (AIC Hurvich bandwidth estimation).
        Default is 'cv_ls'. User specified bandwidth must have as many
        entries as the number of variables.
    ckertype : str, optional
        The kernel used for the continuous variables.
    okertype : str, optional
        The kernel used for the ordered discrete variables.
    ukertype : str, optional
        The kernel used for the unordered discrete variables.
    defaults : EstimatorSettings instance, optional
        The default values for the efficient bandwidth estimation.

    Attributes
    ----------
    bw : array_like
        The bandwidth parameters.
    """

    def __init__(self, endog, exog, var_type, reg_type='ll', bw='cv_ls', ckertype='gaussian', okertype='wangryzin', ukertype='aitchisonaitken', defaults=None):
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
        if not isinstance(bw, str):
            bw = np.asarray(bw)
            if len(bw) != self.k_vars:
                raise ValueError('bw must have the same dimension as the number of variables.')
        if not self.efficient:
            self.bw = self._compute_reg_bw(bw)
        else:
            self.bw = self._compute_efficient(bw)

    def _compute_reg_bw(self, bw):
        if not isinstance(bw, str):
            self._bw_method = 'user-specified'
            return np.asarray(bw)
        else:
            self._bw_method = bw
            if bw == 'cv_ls':
                res = self.cv_loo
            else:
                res = self.aic_hurvich
            X = np.std(self.exog, axis=0)
            h0 = 1.06 * X * self.nobs ** (-1.0 / (4 + np.size(self.exog, axis=1)))
            func = self.est[self.reg_type]
            bw_estimated = optimize.fmin(res, x0=h0, args=(func,), maxiter=1000.0, maxfun=1000.0, disp=0)
            return bw_estimated

    def _est_loc_linear(self, bw, endog, exog, data_predict):
        """
        Local linear estimator of g(x) in the regression ``y = g(x) + e``.

        Parameters
        ----------
        bw : array_like
            Vector of bandwidth value(s).
        endog : 1D array_like
            The dependent variable.
        exog : 1D or 2D array_like
            The independent variable(s).
        data_predict : 1D array_like of length K, where K is the number of variables.
            The point at which the density is estimated.

        Returns
        -------
        D_x : array_like
            The value of the conditional mean at `data_predict`.

        Notes
        -----
        See p. 81 in [1] and p.38 in [2] for the formulas.
        Unlike other methods, this one requires that `data_predict` be 1D.
        """
        nobs, k_vars = exog.shape
        ker = gpke(bw, data=exog, data_predict=data_predict, var_type=self.var_type, ckertype=self.ckertype, ukertype=self.ukertype, okertype=self.okertype, tosum=False) / float(nobs)
        ker = ker[:, np.newaxis]
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

    def _est_loc_constant(self, bw, endog, exog, data_predict):
        """
        Local constant estimator of g(x) in the regression
        y = g(x) + e

        Parameters
        ----------
        bw : array_like
            Array of bandwidth value(s).
        endog : 1D array_like
            The dependent variable.
        exog : 1D or 2D array_like
            The independent variable(s).
        data_predict : 1D or 2D array_like
            The point(s) at which the density is estimated.

        Returns
        -------
        G : ndarray
            The value of the conditional mean at `data_predict`.
        B_x : ndarray
            The marginal effects.
        """
        ker_x = gpke(bw, data=exog, data_predict=data_predict, var_type=self.var_type, ckertype=self.ckertype, ukertype=self.ukertype, okertype=self.okertype, tosum=False)
        ker_x = np.reshape(ker_x, np.shape(endog))
        G_numer = (ker_x * endog).sum(axis=0)
        G_denom = ker_x.sum(axis=0)
        G = G_numer / G_denom
        nobs = exog.shape[0]
        f_x = G_denom / float(nobs)
        ker_xc = gpke(bw, data=exog, data_predict=data_predict, var_type=self.var_type, ckertype='d_gaussian', tosum=False)
        ker_xc = ker_xc[:, np.newaxis]
        d_mx = -(endog * ker_xc).sum(axis=0) / float(nobs)
        d_fx = -ker_xc.sum(axis=0) / float(nobs)
        B_x = d_mx / f_x - G * d_fx / f_x
        B_x = (G_numer * d_fx - G_denom * d_mx) / G_denom ** 2
        return (G, B_x)

    def aic_hurvich(self, bw, func=None):
        """
        Computes the AIC Hurvich criteria for the estimation of the bandwidth.

        Parameters
        ----------
        bw : str or array_like
            See the ``bw`` parameter of `KernelReg` for details.

        Returns
        -------
        aic : ndarray
            The AIC Hurvich criteria, one element for each variable.
        func : None
            Unused here, needed in signature because it's used in `cv_loo`.

        References
        ----------
        See ch.2 in [1] and p.35 in [2].
        """
        H = np.empty((self.nobs, self.nobs))
        for j in range(self.nobs):
            H[:, j] = gpke(bw, data=self.exog, data_predict=self.exog[j, :], ckertype=self.ckertype, ukertype=self.ukertype, okertype=self.okertype, var_type=self.var_type, tosum=False)
        denom = H.sum(axis=1)
        H = H / denom
        gx = KernelReg(endog=self.endog, exog=self.exog, var_type=self.var_type, reg_type=self.reg_type, bw=bw, defaults=EstimatorSettings(efficient=False)).fit()[0]
        gx = np.reshape(gx, (self.nobs, 1))
        sigma = ((self.endog - gx) ** 2).sum(axis=0) / float(self.nobs)
        frac = (1 + np.trace(H) / float(self.nobs)) / (1 - (np.trace(H) + 2) / float(self.nobs))
        aic = np.log(sigma) + frac
        return aic

    def cv_loo(self, bw, func):
        """
        The cross-validation function with leave-one-out estimator.

        Parameters
        ----------
        bw : array_like
            Vector of bandwidth values.
        func : callable function
            Returns the estimator of g(x).  Can be either ``_est_loc_constant``
            (local constant) or ``_est_loc_linear`` (local_linear).

        Returns
        -------
        L : float
            The value of the CV function.

        Notes
        -----
        Calculates the cross-validation least-squares function. This function
        is minimized by compute_bw to calculate the optimal value of `bw`.

        For details see p.35 in [2]

        .. math:: CV(h)=n^{-1}\\sum_{i=1}^{n}(Y_{i}-g_{-i}(X_{i}))^{2}

        where :math:`g_{-i}(X_{i})` is the leave-one-out estimator of g(X)
        and :math:`h` is the vector of bandwidths
        """
        LOO_X = LeaveOneOut(self.exog)
        LOO_Y = LeaveOneOut(self.endog).__iter__()
        L = 0
        for ii, X_not_i in enumerate(LOO_X):
            Y = next(LOO_Y)
            G = func(bw, endog=Y, exog=-X_not_i, data_predict=-self.exog[ii, :])[0]
            L += (self.endog[ii] - G) ** 2
        return L / self.nobs

    def r_squared(self):
        """
        Returns the R-Squared for the nonparametric regression.

        Notes
        -----
        For more details see p.45 in [2]
        The R-Squared is calculated by:

        .. math:: R^{2}=\\frac{\\left[\\sum_{i=1}^{n}
            (Y_{i}-\\bar{y})(\\hat{Y_{i}}-\\bar{y}\\right]^{2}}{\\sum_{i=1}^{n}
            (Y_{i}-\\bar{y})^{2}\\sum_{i=1}^{n}(\\hat{Y_{i}}-\\bar{y})^{2}},

        where :math:`\\hat{Y_{i}}` is the mean calculated in `fit` at the exog
        points.
        """
        Y = np.squeeze(self.endog)
        Yhat = self.fit()[0]
        Y_bar = np.mean(Yhat)
        R2_numer = ((Y - Y_bar) * (Yhat - Y_bar)).sum() ** 2
        R2_denom = ((Y - Y_bar) ** 2).sum(axis=0) * ((Yhat - Y_bar) ** 2).sum(axis=0)
        return R2_numer / R2_denom

    def fit(self, data_predict=None):
        """
        Returns the mean and marginal effects at the `data_predict` points.

        Parameters
        ----------
        data_predict : array_like, optional
            Points at which to return the mean and marginal effects.  If not
            given, ``data_predict == exog``.

        Returns
        -------
        mean : ndarray
            The regression result for the mean (i.e. the actual curve).
        mfx : ndarray
            The marginal effects, i.e. the partial derivatives of the mean.
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
            mean_mfx = func(self.bw, self.endog, self.exog, data_predict=data_predict[i, :])
            mean[i] = np.squeeze(mean_mfx[0])
            mfx_c = np.squeeze(mean_mfx[1])
            mfx[i, :] = mfx_c
        return (mean, mfx)

    def sig_test(self, var_pos, nboot=50, nested_res=25, pivot=False):
        """
        Significance test for the variables in the regression.

        Parameters
        ----------
        var_pos : sequence
            The position of the variable in exog to be tested.

        Returns
        -------
        sig : str
            The level of significance:

                - `*` : at 90% confidence level
                - `**` : at 95% confidence level
                - `***` : at 99* confidence level
                - "Not Significant" : if not significant
        """
        var_pos = np.asarray(var_pos)
        ix_cont, ix_ord, ix_unord = _get_type_pos(self.var_type)
        if np.any(ix_cont[var_pos]):
            if np.any(ix_ord[var_pos]) or np.any(ix_unord[var_pos]):
                raise ValueError('Discrete variable in hypothesis. Must be continuous')
            Sig = TestRegCoefC(self, var_pos, nboot, nested_res, pivot)
        else:
            Sig = TestRegCoefD(self, var_pos, nboot)
        return Sig.sig

    def __repr__(self):
        """Provide something sane to print."""
        rpr = 'KernelReg instance\n'
        rpr += 'Number of variables: k_vars = ' + str(self.k_vars) + '\n'
        rpr += 'Number of samples:   N = ' + str(self.nobs) + '\n'
        rpr += 'Variable types:      ' + self.var_type + '\n'
        rpr += 'BW selection method: ' + self._bw_method + '\n'
        rpr += 'Estimator type: ' + self.reg_type + '\n'
        return rpr

    def _get_class_vars_type(self):
        """Helper method to be able to pass needed vars to _compute_subset."""
        class_type = 'KernelReg'
        class_vars = (self.var_type, self.k_vars, self.reg_type)
        return (class_type, class_vars)

    def _compute_dispersion(self, data):
        """
        Computes the measure of dispersion.

        The minimum of the standard deviation and interquartile range / 1.349

        References
        ----------
        See the user guide for the np package in R.
        In the notes on bwscaling option in npreg, npudens, npcdens there is
        a discussion on the measure of dispersion
        """
        data = data[:, 1:]
        return _compute_min_std_IQR(data)
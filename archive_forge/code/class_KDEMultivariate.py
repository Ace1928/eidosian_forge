import numpy as np
from . import kernels
from ._kernel_base import GenericKDE, EstimatorSettings, gpke, \
class KDEMultivariate(GenericKDE):
    """
    Multivariate kernel density estimator.

    This density estimator can handle univariate as well as multivariate data,
    including mixed continuous / ordered discrete / unordered discrete data.
    It also provides cross-validated bandwidth selection methods (least
    squares, maximum likelihood).

    Parameters
    ----------
    data : list of ndarrays or 2-D ndarray
        The training data for the Kernel Density Estimation, used to determine
        the bandwidth(s).  If a 2-D array, should be of shape
        (num_observations, num_variables).  If a list, each list element is a
        separate observation.
    var_type : str
        The type of the variables:

            - c : continuous
            - u : unordered (discrete)
            - o : ordered (discrete)

        The string should contain a type specifier for each variable, so for
        example ``var_type='ccuo'``.
    bw : array_like or str, optional
        If an array, it is a fixed user-specified bandwidth.  If a string,
        should be one of:

            - normal_reference: normal reference rule of thumb (default)
            - cv_ml: cross validation maximum likelihood
            - cv_ls: cross validation least squares

    defaults : EstimatorSettings instance, optional
        The default values for (efficient) bandwidth estimation.

    Attributes
    ----------
    bw : array_like
        The bandwidth parameters.

    See Also
    --------
    KDEMultivariateConditional

    Examples
    --------
    >>> import statsmodels.api as sm
    >>> nobs = 300
    >>> np.random.seed(1234)  # Seed random generator
    >>> c1 = np.random.normal(size=(nobs,1))
    >>> c2 = np.random.normal(2, 1, size=(nobs,1))

    Estimate a bivariate distribution and display the bandwidth found:

    >>> dens_u = sm.nonparametric.KDEMultivariate(data=[c1,c2],
    ...     var_type='cc', bw='normal_reference')
    >>> dens_u.bw
    array([ 0.39967419,  0.38423292])
    """

    def __init__(self, data, var_type, bw=None, defaults=None):
        self.var_type = var_type
        self.k_vars = len(self.var_type)
        self.data = _adjust_shape(data, self.k_vars)
        self.data_type = var_type
        self.nobs, self.k_vars = np.shape(self.data)
        if self.nobs <= self.k_vars:
            raise ValueError('The number of observations must be larger than the number of variables.')
        defaults = EstimatorSettings() if defaults is None else defaults
        self._set_defaults(defaults)
        if not self.efficient:
            self.bw = self._compute_bw(bw)
        else:
            self.bw = self._compute_efficient(bw)

    def __repr__(self):
        """Provide something sane to print."""
        rpr = 'KDE instance\n'
        rpr += 'Number of variables: k_vars = ' + str(self.k_vars) + '\n'
        rpr += 'Number of samples:   nobs = ' + str(self.nobs) + '\n'
        rpr += 'Variable types:      ' + self.var_type + '\n'
        rpr += 'BW selection method: ' + self._bw_method + '\n'
        return rpr

    def loo_likelihood(self, bw, func=lambda x: x):
        """
        Returns the leave-one-out likelihood function.

        The leave-one-out likelihood function for the unconditional KDE.

        Parameters
        ----------
        bw : array_like
            The value for the bandwidth parameter(s).
        func : callable, optional
            Function to transform the likelihood values (before summing); for
            the log likelihood, use ``func=np.log``.  Default is ``f(x) = x``.

        Notes
        -----
        The leave-one-out kernel estimator of :math:`f_{-i}` is:

        .. math:: f_{-i}(X_{i})=\\frac{1}{(n-1)h}
                    \\sum_{j=1,j\\neq i}K_{h}(X_{i},X_{j})

        where :math:`K_{h}` represents the generalized product kernel
        estimator:

        .. math:: K_{h}(X_{i},X_{j}) =
            \\prod_{s=1}^{q}h_{s}^{-1}k\\left(\\frac{X_{is}-X_{js}}{h_{s}}\\right)
        """
        LOO = LeaveOneOut(self.data)
        L = 0
        for i, X_not_i in enumerate(LOO):
            f_i = gpke(bw, data=-X_not_i, data_predict=-self.data[i, :], var_type=self.var_type)
            L += func(f_i)
        return -L

    def pdf(self, data_predict=None):
        """
        Evaluate the probability density function.

        Parameters
        ----------
        data_predict : array_like, optional
            Points to evaluate at.  If unspecified, the training data is used.

        Returns
        -------
        pdf_est : array_like
            Probability density function evaluated at `data_predict`.

        Notes
        -----
        The probability density is given by the generalized product kernel
        estimator:

        .. math:: K_{h}(X_{i},X_{j}) =
            \\prod_{s=1}^{q}h_{s}^{-1}k\\left(\\frac{X_{is}-X_{js}}{h_{s}}\\right)
        """
        if data_predict is None:
            data_predict = self.data
        else:
            data_predict = _adjust_shape(data_predict, self.k_vars)
        pdf_est = []
        for i in range(np.shape(data_predict)[0]):
            pdf_est.append(gpke(self.bw, data=self.data, data_predict=data_predict[i, :], var_type=self.var_type) / self.nobs)
        pdf_est = np.squeeze(pdf_est)
        return pdf_est

    def cdf(self, data_predict=None):
        """
        Evaluate the cumulative distribution function.

        Parameters
        ----------
        data_predict : array_like, optional
            Points to evaluate at.  If unspecified, the training data is used.

        Returns
        -------
        cdf_est : array_like
            The estimate of the cdf.

        Notes
        -----
        See https://en.wikipedia.org/wiki/Cumulative_distribution_function
        For more details on the estimation see Ref. [5] in module docstring.

        The multivariate CDF for mixed data (continuous and ordered/unordered
        discrete) is estimated by:

        .. math::

            F(x^{c},x^{d})=n^{-1}\\sum_{i=1}^{n}\\left[G(\\frac{x^{c}-X_{i}}{h})\\sum_{u\\leq x^{d}}L(X_{i}^{d},x_{i}^{d}, \\lambda)\\right]

        where G() is the product kernel CDF estimator for the continuous
        and L() for the discrete variables.

        Used bandwidth is ``self.bw``.
        """
        if data_predict is None:
            data_predict = self.data
        else:
            data_predict = _adjust_shape(data_predict, self.k_vars)
        cdf_est = []
        for i in range(np.shape(data_predict)[0]):
            cdf_est.append(gpke(self.bw, data=self.data, data_predict=data_predict[i, :], var_type=self.var_type, ckertype='gaussian_cdf', ukertype='aitchisonaitken_cdf', okertype='wangryzin_cdf') / self.nobs)
        cdf_est = np.squeeze(cdf_est)
        return cdf_est

    def imse(self, bw):
        """
        Returns the Integrated Mean Square Error for the unconditional KDE.

        Parameters
        ----------
        bw : array_like
            The bandwidth parameter(s).

        Returns
        -------
        CV : float
            The cross-validation objective function.

        Notes
        -----
        See p. 27 in [1]_ for details on how to handle the multivariate
        estimation with mixed data types see p.6 in [2]_.

        The formula for the cross-validation objective function is:

        .. math:: CV=\\frac{1}{n^{2}}\\sum_{i=1}^{n}\\sum_{j=1}^{N}
            \\bar{K}_{h}(X_{i},X_{j})-\\frac{2}{n(n-1)}\\sum_{i=1}^{n}
            \\sum_{j=1,j\\neq i}^{N}K_{h}(X_{i},X_{j})

        Where :math:`\\bar{K}_{h}` is the multivariate product convolution
        kernel (consult [2]_ for mixed data types).

        References
        ----------
        .. [1] Racine, J., Li, Q. Nonparametric econometrics: theory and
                practice. Princeton University Press. (2007)
        .. [2] Racine, J., Li, Q. "Nonparametric Estimation of Distributions
                with Categorical and Continuous Data." Working Paper. (2000)
        """
        F = 0
        kertypes = dict(c=kernels.gaussian_convolution, o=kernels.wang_ryzin_convolution, u=kernels.aitchison_aitken_convolution)
        nobs = self.nobs
        data = -self.data
        var_type = self.var_type
        ix_cont = np.array([c == 'c' for c in var_type])
        _bw_cont_product = bw[ix_cont].prod()
        Kval = np.empty(data.shape)
        for i in range(nobs):
            for ii, vtype in enumerate(var_type):
                Kval[:, ii] = kertypes[vtype](bw[ii], data[:, ii], data[i, ii])
            dens = Kval.prod(axis=1) / _bw_cont_product
            k_bar_sum = dens.sum(axis=0)
            F += k_bar_sum
        kertypes = dict(c=kernels.gaussian, o=kernels.wang_ryzin, u=kernels.aitchison_aitken)
        LOO = LeaveOneOut(self.data)
        L = 0
        Kval = np.empty((data.shape[0] - 1, data.shape[1]))
        for i, X_not_i in enumerate(LOO):
            for ii, vtype in enumerate(var_type):
                Kval[:, ii] = kertypes[vtype](bw[ii], -X_not_i[:, ii], data[i, ii])
            dens = Kval.prod(axis=1) / _bw_cont_product
            L += dens.sum(axis=0)
        return F / nobs ** 2 - 2 * L / (nobs * (nobs - 1))

    def _get_class_vars_type(self):
        """Helper method to be able to pass needed vars to _compute_subset."""
        class_type = 'KDEMultivariate'
        class_vars = (self.var_type,)
        return (class_type, class_vars)
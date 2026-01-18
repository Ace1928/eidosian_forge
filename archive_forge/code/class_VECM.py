from collections import defaultdict
import numpy as np
from numpy import hstack, vstack
from numpy.linalg import inv, svd
import scipy
import scipy.stats
from statsmodels.iolib.summary import Summary
from statsmodels.iolib.table import SimpleTable
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.sm_exceptions import HypothesisTestWarning
from statsmodels.tools.validation import string_like
import statsmodels.tsa.base.tsa_model as tsbase
from statsmodels.tsa.coint_tables import c_sja, c_sjt
from statsmodels.tsa.tsatools import duplication_matrix, lagmat, vec
from statsmodels.tsa.vector_ar.hypothesis_test_results import (
import statsmodels.tsa.vector_ar.irf as irf
import statsmodels.tsa.vector_ar.plotting as plot
from statsmodels.tsa.vector_ar.util import get_index, seasonal_dummies
from statsmodels.tsa.vector_ar.var_model import (
class VECM(tsbase.TimeSeriesModel):
    """
    Class representing a Vector Error Correction Model (VECM).

    A VECM(:math:`k_{ar}-1`) has the following form

    .. math:: \\Delta y_t = \\Pi y_{t-1} + \\Gamma_1 \\Delta y_{t-1} + \\ldots + \\Gamma_{k_{ar}-1} \\Delta y_{t-k_{ar}+1} + u_t

    where

    .. math:: \\Pi = \\alpha \\beta'

    as described in chapter 7 of [1]_.

    Parameters
    ----------
    endog : array_like (nobs_tot x neqs)
        2-d endogenous response variable.
    exog : ndarray (nobs_tot x neqs) or None
        Deterministic terms outside the cointegration relation.
    exog_coint : ndarray (nobs_tot x neqs) or None
        Deterministic terms inside the cointegration relation.
    dates : array_like of datetime, optional
        See :class:`statsmodels.tsa.base.tsa_model.TimeSeriesModel` for more
        information.
    freq : str, optional
        See :class:`statsmodels.tsa.base.tsa_model.TimeSeriesModel` for more
        information.
    missing : str, optional
        See :class:`statsmodels.base.model.Model` for more information.
    k_ar_diff : int
        Number of lagged differences in the model. Equals :math:`k_{ar} - 1` in
        the formula above.
    coint_rank : int
        Cointegration rank, equals the rank of the matrix :math:`\\Pi` and the
        number of columns of :math:`\\alpha` and :math:`\\beta`.
    deterministic : str {``"n"``, ``"co"``, ``"ci"``, ``"lo"``, ``"li"``}
        * ``"n"`` - no deterministic terms
        * ``"co"`` - constant outside the cointegration relation
        * ``"ci"`` - constant within the cointegration relation
        * ``"lo"`` - linear trend outside the cointegration relation
        * ``"li"`` - linear trend within the cointegration relation

        Combinations of these are possible (e.g. ``"cili"`` or ``"colo"`` for
        linear trend with intercept). When using a constant term you have to
        choose whether you want to restrict it to the cointegration relation
        (i.e. ``"ci"``) or leave it unrestricted (i.e. ``"co"``). Do not use
        both ``"ci"`` and ``"co"``. The same applies for ``"li"`` and ``"lo"``
        when using a linear term. See the Notes-section for more information.
    seasons : int, default: 0
        Number of periods in a seasonal cycle. 0 means no seasons.
    first_season : int, default: 0
        Season of the first observation.

    Notes
    -----
    A VECM(:math:`k_{ar} - 1`) with deterministic terms has the form

    .. math::

       \\Delta y_t = \\alpha \\begin{pmatrix}\\beta' & \\eta'\\end{pmatrix} \\begin{pmatrix}y_{t-1}\\\\D^{co}_{t-1}\\end{pmatrix} + \\Gamma_1 \\Delta y_{t-1} + \\dots + \\Gamma_{k_{ar}-1} \\Delta y_{t-k_{ar}+1} + C D_t + u_t.

    In :math:`D^{co}_{t-1}` we have the deterministic terms which are inside
    the cointegration relation (or restricted to the cointegration relation).
    :math:`\\eta` is the corresponding estimator. To pass a deterministic term
    inside the cointegration relation, we can use the `exog_coint` argument.
    For the two special cases of an intercept and a linear trend there exists
    a simpler way to declare these terms: we can pass ``"ci"`` and ``"li"``
    respectively to the `deterministic` argument. So for an intercept inside
    the cointegration relation we can either pass ``"ci"`` as `deterministic`
    or `np.ones(len(data))` as `exog_coint` if `data` is passed as the
    `endog` argument. This ensures that :math:`D_{t-1}^{co} = 1` for all
    :math:`t`.

    We can also use deterministic terms outside the cointegration relation.
    These are defined in :math:`D_t` in the formula above with the
    corresponding estimators in the matrix :math:`C`. We specify such terms by
    passing them to the `exog` argument. For an intercept and/or linear trend
    we again have the possibility to use `deterministic` alternatively. For
    an intercept we pass ``"co"`` and for a linear trend we pass ``"lo"`` where
    the `o` stands for `outside`.

    The following table shows the five cases considered in [2]_. The last
    column indicates which string to pass to the `deterministic` argument for
    each of these cases.

    ====  ===============================  ===================================  =============
    Case  Intercept                        Slope of the linear trend            `deterministic`
    ====  ===============================  ===================================  =============
    I     0                                0                                    ``"n"``
    II    :math:`- \\alpha \\beta^T \\mu`     0                                    ``"ci"``
    III   :math:`\\neq 0`                   0                                    ``"co"``
    IV    :math:`\\neq 0`                   :math:`- \\alpha \\beta^T \\gamma`      ``"coli"``
    V     :math:`\\neq 0`                   :math:`\\neq 0`                       ``"colo"``
    ====  ===============================  ===================================  =============

    References
    ----------
    .. [1] Lütkepohl, H. 2005. *New Introduction to Multiple Time Series Analysis*. Springer.

    .. [2] Johansen, S. 1995. *Likelihood-Based Inference in Cointegrated *
           *Vector Autoregressive Models*. Oxford University Press.
    """

    def __init__(self, endog, exog=None, exog_coint=None, dates=None, freq=None, missing='none', k_ar_diff=1, coint_rank=1, deterministic='n', seasons=0, first_season=0):
        super().__init__(endog, exog, dates, freq, missing=missing)
        if exog_coint is not None and (not exog_coint.shape[0] == endog.shape[0]):
            raise ValueError('exog_coint must have as many rows as enodg_tot!')
        if self.endog.ndim == 1:
            raise ValueError('Only gave one variable to VECM')
        self.y = self.endog.T
        self.exog_coint = exog_coint
        self.neqs = self.endog.shape[1]
        self.k_ar = k_ar_diff + 1
        self.k_ar_diff = k_ar_diff
        self.coint_rank = coint_rank
        self.deterministic = deterministic
        self.seasons = seasons
        self.first_season = first_season
        self.load_coef_repr = 'ec'

    def fit(self, method='ml'):
        """
        Estimates the parameters of a VECM.

        The estimation procedure is described on pp. 269-304 in [1]_.

        Parameters
        ----------
        method : str {"ml"}, default: "ml"
            Estimation method to use. "ml" stands for Maximum Likelihood.

        Returns
        -------
        est : :class:`VECMResults`

        References
        ----------
        .. [1] Lütkepohl, H. 2005. *New Introduction to Multiple Time Series Analysis*. Springer.
        """
        if method == 'ml':
            return self._estimate_vecm_ml()
        else:
            raise ValueError('{} not recognized, must be among {}'.format(method, 'ml'))

    def _estimate_vecm_ml(self):
        y_1_T, delta_y_1_T, y_lag1, delta_x = _endog_matrices(self.y, self.exog, self.exog_coint, self.k_ar_diff, self.deterministic, self.seasons, self.first_season)
        T = y_1_T.shape[1]
        s00, s01, s10, s11, s11_, _, v = _sij(delta_x, delta_y_1_T, y_lag1)
        beta_tilde = v[:, :self.coint_rank].T.dot(s11_).T
        beta_tilde = np.real_if_close(beta_tilde)
        beta_tilde = np.dot(beta_tilde, inv(beta_tilde[:self.coint_rank]))
        alpha_tilde = s01.dot(beta_tilde).dot(inv(beta_tilde.T.dot(s11).dot(beta_tilde)))
        gamma_tilde = (delta_y_1_T - alpha_tilde.dot(beta_tilde.T).dot(y_lag1)).dot(delta_x.T).dot(inv(np.dot(delta_x, delta_x.T)))
        temp = delta_y_1_T - alpha_tilde.dot(beta_tilde.T).dot(y_lag1) - gamma_tilde.dot(delta_x)
        sigma_u_tilde = temp.dot(temp.T) / T
        return VECMResults(self.y, self.exog, self.exog_coint, self.k_ar, self.coint_rank, alpha_tilde, beta_tilde, gamma_tilde, sigma_u_tilde, deterministic=self.deterministic, seasons=self.seasons, delta_y_1_T=delta_y_1_T, y_lag1=y_lag1, delta_x=delta_x, model=self, names=self.endog_names, dates=self.data.dates, first_season=self.first_season)

    @property
    def _lagged_param_names(self):
        """
        Returns parameter names (for Gamma and deterministics) for the summary.

        Returns
        -------
        param_names : list of str
            Returns a list of parameter names for the lagged endogenous
            parameters which are called :math:`\\Gamma` in [1]_
            (see chapter 6).
            If present in the model, also names for deterministic terms outside
            the cointegration relation are returned. They name the elements of
            the matrix C in [1]_ (p. 299).

        References
        ----------
        .. [1] Lütkepohl, H. 2005. *New Introduction to Multiple Time Series Analysis*. Springer.
        """
        param_names = []
        if 'co' in self.deterministic:
            param_names += ['const.%s' % n for n in self.endog_names]
        if self.seasons > 0:
            param_names += ['season%d.%s' % (s, n) for s in range(1, self.seasons) for n in self.endog_names]
        if 'lo' in self.deterministic:
            param_names += ['lin_trend.%s' % n for n in self.endog_names]
        if self.exog is not None:
            param_names += ['exog%d.%s' % (exog_no, n) for exog_no in range(1, self.exog.shape[1] + 1) for n in self.endog_names]
        param_names += ['L%d.%s.%s' % (i + 1, n1, n2) for n2 in self.endog_names for i in range(self.k_ar_diff) for n1 in self.endog_names]
        return param_names

    @property
    def _load_coef_param_names(self):
        """
        Returns parameter names (for alpha) for the summary.

        Returns
        -------
        param_names : list of str
            Returns a list of parameter names for the loading coefficients
            which are called :math:`\\alpha` in [1]_ (see chapter 6).

        References
        ----------
        .. [1] Lütkepohl, H. 2005. *New Introduction to Multiple Time Series Analysis*. Springer.
        """
        param_names = []
        if self.coint_rank == 0:
            return None
        param_names += [self.load_coef_repr + '%d.%s' % (i + 1, self.endog_names[j]) for j in range(self.neqs) for i in range(self.coint_rank)]
        return param_names

    @property
    def _coint_param_names(self):
        """
        Returns parameter names (for beta and deterministics) for the summary.

        Returns
        -------
        param_names : list of str
            Returns a list of parameter names for the cointegration matrix
            as well as deterministic terms inside the cointegration relation
            (if present in the model).
        """
        param_names = []
        param_names += [('beta.%d.' + self.load_coef_repr + '%d') % (j + 1, i + 1) for i in range(self.coint_rank) for j in range(self.neqs)]
        if 'ci' in self.deterministic:
            param_names += ['const.' + self.load_coef_repr + '%d' % (i + 1) for i in range(self.coint_rank)]
        if 'li' in self.deterministic:
            param_names += ['lin_trend.' + self.load_coef_repr + '%d' % (i + 1) for i in range(self.coint_rank)]
        if self.exog_coint is not None:
            param_names += ['exog_coint%d.%s' % (n + 1, exog_no) for exog_no in range(1, self.exog_coint.shape[1] + 1) for n in range(self.neqs)]
        return param_names
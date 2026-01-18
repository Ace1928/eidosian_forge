import warnings
from statsmodels.compat.pandas import Appender
from statsmodels.compat.python import lzip
from collections import defaultdict
import numpy as np
from statsmodels.graphics._regressionplots_doc import _plot_influence_doc
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.tools import maybe_unwrap_results
class OLSInfluence(_BaseInfluenceMixin):
    """class to calculate outlier and influence measures for OLS result

    Parameters
    ----------
    results : RegressionResults
        currently assumes the results are from an OLS regression

    Notes
    -----
    One part of the results can be calculated without any auxiliary regression
    (some of which have the `_internal` postfix in the name. Other statistics
    require leave-one-observation-out (LOOO) auxiliary regression, and will be
    slower (mainly results with `_external` postfix in the name).
    The auxiliary LOOO regression only the required results are stored.

    Using the LOO measures is currently only recommended if the data set
    is not too large. One possible approach for LOOO measures would be to
    identify possible problem observations with the _internal measures, and
    then run the leave-one-observation-out only with observations that are
    possible outliers. (However, this is not yet available in an automated way.)

    This should be extended to general least squares.

    The leave-one-variable-out (LOVO) auxiliary regression are currently not
    used.
    """

    def __init__(self, results):
        self.results = maybe_unwrap_results(results)
        self.nobs, self.k_vars = results.model.exog.shape
        self.endog = results.model.endog
        self.exog = results.model.exog
        self.resid = results.resid
        self.model_class = results.model.__class__
        self.scale = results.mse_resid
        self.aux_regression_exog = {}
        self.aux_regression_endog = {}

    @cache_readonly
    def hat_matrix_diag(self):
        """Diagonal of the hat_matrix for OLS

        Notes
        -----
        temporarily calculated here, this should go to model class
        """
        return (self.exog * self.results.model.pinv_wexog.T).sum(1)

    @cache_readonly
    def resid_press(self):
        """PRESS residuals
        """
        hii = self.hat_matrix_diag
        return self.resid / (1 - hii)

    @cache_readonly
    def influence(self):
        """Influence measure

        matches the influence measure that gretl reports
        u * h / (1 - h)
        where u are the residuals and h is the diagonal of the hat_matrix
        """
        hii = self.hat_matrix_diag
        return self.resid * hii / (1 - hii)

    @cache_readonly
    def hat_diag_factor(self):
        """Factor of diagonal of hat_matrix used in influence

        this might be useful for internal reuse
        h / (1 - h)
        """
        hii = self.hat_matrix_diag
        return hii / (1 - hii)

    @cache_readonly
    def ess_press(self):
        """Error sum of squares of PRESS residuals
        """
        return np.dot(self.resid_press, self.resid_press)

    @cache_readonly
    def resid_studentized(self):
        """Studentized residuals using variance from OLS

        alias for resid_studentized_internal for compatibility with
        MLEInfluence this uses sigma from original estimate and does
        not require leave one out loop
        """
        return self.resid_studentized_internal

    @cache_readonly
    def resid_studentized_internal(self):
        """Studentized residuals using variance from OLS

        this uses sigma from original estimate
        does not require leave one out loop
        """
        return self.get_resid_studentized_external(sigma=None)

    @cache_readonly
    def resid_studentized_external(self):
        """Studentized residuals using LOOO variance

        this uses sigma from leave-one-out estimates

        requires leave one out loop for observations
        """
        sigma_looo = np.sqrt(self.sigma2_not_obsi)
        return self.get_resid_studentized_external(sigma=sigma_looo)

    def get_resid_studentized_external(self, sigma=None):
        """calculate studentized residuals

        Parameters
        ----------
        sigma : None or float
            estimate of the standard deviation of the residuals. If None, then
            the estimate from the regression results is used.

        Returns
        -------
        stzd_resid : ndarray
            studentized residuals

        Notes
        -----
        studentized residuals are defined as ::

           resid / sigma / np.sqrt(1 - hii)

        where resid are the residuals from the regression, sigma is an
        estimate of the standard deviation of the residuals, and hii is the
        diagonal of the hat_matrix.
        """
        hii = self.hat_matrix_diag
        if sigma is None:
            sigma2_est = self.scale
            sigma = np.sqrt(sigma2_est)
        return self.resid / sigma / np.sqrt(1 - hii)

    @cache_readonly
    def cooks_distance(self):
        """
        Cooks distance

        Uses original results, no nobs loop

        References
        ----------
        .. [*] Eubank, R. L. (1999). Nonparametric regression and spline
            smoothing. CRC press.
        .. [*] Cook's distance. (n.d.). In Wikipedia. July 2019, from
            https://en.wikipedia.org/wiki/Cook%27s_distance
        """
        hii = self.hat_matrix_diag
        cooks_d2 = self.resid_studentized ** 2 / self.k_vars
        cooks_d2 *= hii / (1 - hii)
        from scipy import stats
        pvals = stats.f.sf(cooks_d2, self.k_vars, self.results.df_resid)
        return (cooks_d2, pvals)

    @cache_readonly
    def dffits_internal(self):
        """dffits measure for influence of an observation

        based on resid_studentized_internal
        uses original results, no nobs loop
        """
        hii = self.hat_matrix_diag
        dffits_ = self.resid_studentized_internal * np.sqrt(hii / (1 - hii))
        dffits_threshold = 2 * np.sqrt(self.k_vars * 1.0 / self.nobs)
        return (dffits_, dffits_threshold)

    @cache_readonly
    def dffits(self):
        """
        dffits measure for influence of an observation

        based on resid_studentized_external,
        uses results from leave-one-observation-out loop

        It is recommended that observations with dffits large than a
        threshold of 2 sqrt{k / n} where k is the number of parameters, should
        be investigated.

        Returns
        -------
        dffits : float
        dffits_threshold : float

        References
        ----------
        `Wikipedia <https://en.wikipedia.org/wiki/DFFITS>`_
        """
        hii = self.hat_matrix_diag
        dffits_ = self.resid_studentized_external * np.sqrt(hii / (1 - hii))
        dffits_threshold = 2 * np.sqrt(self.k_vars * 1.0 / self.nobs)
        return (dffits_, dffits_threshold)

    @cache_readonly
    def dfbetas(self):
        """dfbetas

        uses results from leave-one-observation-out loop
        """
        dfbetas = self.results.params - self.params_not_obsi
        dfbetas /= np.sqrt(self.sigma2_not_obsi[:, None])
        dfbetas /= np.sqrt(np.diag(self.results.normalized_cov_params))
        return dfbetas

    @cache_readonly
    def dfbeta(self):
        """dfbetas

        uses results from leave-one-observation-out loop
        """
        dfbeta = self.results.params - self.params_not_obsi
        return dfbeta

    @cache_readonly
    def sigma2_not_obsi(self):
        """error variance for all LOOO regressions

        This is 'mse_resid' from each auxiliary regression.

        uses results from leave-one-observation-out loop
        """
        return np.asarray(self._res_looo['mse_resid'])

    @property
    def params_not_obsi(self):
        """parameter estimates for all LOOO regressions

        uses results from leave-one-observation-out loop
        """
        return np.asarray(self._res_looo['params'])

    @property
    def det_cov_params_not_obsi(self):
        """determinant of cov_params of all LOOO regressions

        uses results from leave-one-observation-out loop
        """
        return np.asarray(self._res_looo['det_cov_params'])

    @cache_readonly
    def cov_ratio(self):
        """covariance ratio between LOOO and original

        This uses determinant of the estimate of the parameter covariance
        from leave-one-out estimates.
        requires leave one out loop for observations
        """
        cov_ratio = self.det_cov_params_not_obsi / np.linalg.det(self.results.cov_params())
        return cov_ratio

    @cache_readonly
    def resid_var(self):
        """estimate of variance of the residuals

        ::

           sigma2 = sigma2_OLS * (1 - hii)

        where hii is the diagonal of the hat matrix
        """
        return self.scale * (1 - self.hat_matrix_diag)

    @cache_readonly
    def resid_std(self):
        """estimate of standard deviation of the residuals

        See Also
        --------
        resid_var
        """
        return np.sqrt(self.resid_var)

    def _ols_xnoti(self, drop_idx, endog_idx='endog', store=True):
        """regression results from LOVO auxiliary regression with cache


        The result instances are stored, which could use a large amount of
        memory if the datasets are large. There are too many combinations to
        store them all, except for small problems.

        Parameters
        ----------
        drop_idx : int
            index of exog that is dropped from the regression
        endog_idx : 'endog' or int
            If 'endog', then the endogenous variable of the result instance
            is regressed on the exogenous variables, excluding the one at
            drop_idx. If endog_idx is an integer, then the exog with that
            index is regressed with OLS on all other exogenous variables.
            (The latter is the auxiliary regression for the variance inflation
            factor.)

        this needs more thought, memory versus speed
        not yet used in any other parts, not sufficiently tested
        """
        if endog_idx == 'endog':
            stored = self.aux_regression_endog
            if hasattr(stored, drop_idx):
                return stored[drop_idx]
            x_i = self.results.model.endog
        else:
            try:
                self.aux_regression_exog[endog_idx][drop_idx]
            except KeyError:
                pass
            stored = self.aux_regression_exog[endog_idx]
            stored = {}
            x_i = self.exog[:, endog_idx]
        k_vars = self.exog.shape[1]
        mask = np.arange(k_vars) != drop_idx
        x_noti = self.exog[:, mask]
        res = OLS(x_i, x_noti).fit()
        if store:
            stored[drop_idx] = res
        return res

    def _get_drop_vari(self, attributes):
        """
        regress endog on exog without one of the variables

        This uses a k_vars loop, only attributes of the OLS instance are
        stored.

        Parameters
        ----------
        attributes : list[str]
           These are the names of the attributes of the auxiliary OLS results
           instance that are stored and returned.

        not yet used
        """
        from statsmodels.sandbox.tools.cross_val import LeaveOneOut
        endog = self.results.model.endog
        exog = self.exog
        cv_iter = LeaveOneOut(self.k_vars)
        res_loo = defaultdict(list)
        for inidx, outidx in cv_iter:
            for att in attributes:
                res_i = self.model_class(endog, exog[:, inidx]).fit()
                res_loo[att].append(getattr(res_i, att))
        return res_loo

    @cache_readonly
    def _res_looo(self):
        """collect required results from the LOOO loop

        all results will be attached.
        currently only 'params', 'mse_resid', 'det_cov_params' are stored

        regresses endog on exog dropping one observation at a time

        this uses a nobs loop, only attributes of the OLS instance are stored.
        """
        from statsmodels.sandbox.tools.cross_val import LeaveOneOut

        def get_det_cov_params(res):
            return np.linalg.det(res.cov_params())
        endog = self.results.model.endog
        exog = self.results.model.exog
        params = np.zeros(exog.shape, dtype=float)
        mse_resid = np.zeros(endog.shape, dtype=float)
        det_cov_params = np.zeros(endog.shape, dtype=float)
        cv_iter = LeaveOneOut(self.nobs)
        for inidx, outidx in cv_iter:
            res_i = self.model_class(endog[inidx], exog[inidx]).fit()
            params[outidx] = res_i.params
            mse_resid[outidx] = res_i.mse_resid
            det_cov_params[outidx] = get_det_cov_params(res_i)
        return dict(params=params, mse_resid=mse_resid, det_cov_params=det_cov_params)

    def summary_frame(self):
        """
        Creates a DataFrame with all available influence results.

        Returns
        -------
        frame : DataFrame
            A DataFrame with all results.

        Notes
        -----
        The resultant DataFrame contains six variables in addition to the
        DFBETAS. These are:

        * cooks_d : Cook's Distance defined in `Influence.cooks_distance`
        * standard_resid : Standardized residuals defined in
          `Influence.resid_studentized_internal`
        * hat_diag : The diagonal of the projection, or hat, matrix defined in
          `Influence.hat_matrix_diag`
        * dffits_internal : DFFITS statistics using internally Studentized
          residuals defined in `Influence.dffits_internal`
        * dffits : DFFITS statistics using externally Studentized residuals
          defined in `Influence.dffits`
        * student_resid : Externally Studentized residuals defined in
          `Influence.resid_studentized_external`
        """
        from pandas import DataFrame
        data = self.results.model.data
        row_labels = data.row_labels
        beta_labels = ['dfb_' + i for i in data.xnames]
        summary_data = DataFrame(dict(cooks_d=self.cooks_distance[0], standard_resid=self.resid_studentized_internal, hat_diag=self.hat_matrix_diag, dffits_internal=self.dffits_internal[0], student_resid=self.resid_studentized_external, dffits=self.dffits[0]), index=row_labels)
        dfbeta = DataFrame(self.dfbetas, columns=beta_labels, index=row_labels)
        return dfbeta.join(summary_data)

    def summary_table(self, float_fmt='%6.3f'):
        """create a summary table with all influence and outlier measures

        This does currently not distinguish between statistics that can be
        calculated from the original regression results and for which a
        leave-one-observation-out loop is needed

        Returns
        -------
        res : SimpleTable
           SimpleTable instance with the results, can be printed

        Notes
        -----
        This also attaches table_data to the instance.
        """
        table_raw = [('obs', np.arange(self.nobs)), ('endog', self.endog), ('fitted\nvalue', self.results.fittedvalues), ("Cook's\nd", self.cooks_distance[0]), ('student.\nresidual', self.resid_studentized_internal), ('hat diag', self.hat_matrix_diag), ('dffits \ninternal', self.dffits_internal[0]), ('ext.stud.\nresidual', self.resid_studentized_external), ('dffits', self.dffits[0])]
        colnames, data = lzip(*table_raw)
        data = np.column_stack(data)
        self.table_data = data
        from copy import deepcopy
        from statsmodels.iolib.table import SimpleTable, default_html_fmt
        from statsmodels.iolib.tableformatting import fmt_base
        fmt = deepcopy(fmt_base)
        fmt_html = deepcopy(default_html_fmt)
        fmt['data_fmts'] = ['%4d'] + [float_fmt] * (data.shape[1] - 1)
        return SimpleTable(data, headers=colnames, txt_fmt=fmt, html_fmt=fmt_html)
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
class MLEInfluence(_BaseInfluenceMixin):
    """Global Influence and outlier measures (experimental)

    Parameters
    ----------
    results : instance of results class
        This only works for model and results classes that have the necessary
        helper methods.
    other arguments :
        Those are only available to override default behavior and are used
        instead of the corresponding attribute of the results class.
        By default resid_pearson is used as resid.

    Attributes
    ----------
    hat_matrix_diag (hii) : This is the generalized leverage computed as the
        local derivative of fittedvalues (predicted mean) with respect to the
        observed response for each observation.
        Not available for ZeroInflated models because of nondifferentiability.
    d_params : Change in parameters computed with one Newton step using the
        full Hessian corrected by division by (1 - hii).
        If hat_matrix_diag is not available, then the division by (1 - hii) is
        not included.
    dbetas : change in parameters divided by the standard error of parameters
        from the full model results, ``bse``.
    cooks_distance : quadratic form for change in parameters weighted by
        ``cov_params`` from the full model divided by the number of variables.
        It includes p-values based on the F-distribution which are only
        approximate outside of linear Gaussian models.
    resid_studentized : In the general MLE case resid_studentized are
        computed from the score residuals scaled by hessian factor and
        leverage. This does not use ``cov_params``.
    d_fittedvalues : local change of expected mean given the change in the
        parameters as computed in ``d_params``.
    d_fittedvalues_scaled : same as d_fittedvalues but scaled by the standard
        errors of a predicted mean of the response.
    params_one : is the one step parameter estimate computed as ``params``
        from the full sample minus ``d_params``.

    Notes
    -----
    MLEInfluence uses generic definitions based on maximum likelihood models.

    MLEInfluence produces the same results as GLMInfluence for canonical
    links (verified for GLM Binomial, Poisson and Gaussian). There will be
    some differences for non-canonical links or if a robust cov_type is used.
    For example, the generalized leverage differs from the definition of the
    GLM hat matrix in the case of Probit, which corresponds to family
    Binomial with a non-canonical link.

    The extension to non-standard models, e.g. multi-link model like
    BetaModel and the ZeroInflated models is still experimental and might still
    change.
    Additonally, ZeroInflated and some threshold models have a
    nondifferentiability in the generalized leverage. How this case is treated
    might also change.

    Warning: This does currently not work for constrained or penalized models,
    e.g. models estimated with fit_constrained or fit_regularized.

    This has not yet been tested for correctness when offset or exposure
    are used, although they should be supported by the code.

    status: experimental,
    This class will need changes to support different kinds of models, e.g.
    extra parameters in discrete.NegativeBinomial or two-part models like
    ZeroInflatedPoisson.
    """

    def __init__(self, results, resid=None, endog=None, exog=None, hat_matrix_diag=None, cov_params=None, scale=None):
        self.results = results = maybe_unwrap_results(results)
        self.nobs, self.k_vars = results.model.exog.shape
        self.k_params = np.size(results.params)
        self.endog = endog if endog is not None else results.model.endog
        self.exog = exog if exog is not None else results.model.exog
        self.scale = scale if scale is not None else results.scale
        if resid is not None:
            self.resid = resid
        else:
            self.resid = getattr(results, 'resid_pearson', None)
            if self.resid is not None:
                self.resid = self.resid / np.sqrt(self.scale)
        self.cov_params = cov_params if cov_params is not None else results.cov_params()
        self.model_class = results.model.__class__
        self.hessian = self.results.model.hessian(self.results.params)
        self.score_obs = self.results.model.score_obs(self.results.params)
        if hat_matrix_diag is not None:
            self._hat_matrix_diag = hat_matrix_diag

    @cache_readonly
    def hat_matrix_diag(self):
        """Diagonal of the generalized leverage

        This is the analogue of the hat matrix diagonal for general MLE.
        """
        if hasattr(self, '_hat_matrix_diag'):
            return self._hat_matrix_diag
        try:
            dsdy = self.results.model._deriv_score_obs_dendog(self.results.params)
        except NotImplementedError:
            dsdy = None
        if dsdy is None:
            warnings.warn('hat matrix is not available, missing derivatives', UserWarning)
            return None
        dmu_dp = self.results.model._deriv_mean_dparams(self.results.params)
        h = (dmu_dp * np.linalg.solve(-self.hessian, dsdy.T).T).sum(1)
        return h

    @cache_readonly
    def hat_matrix_exog_diag(self):
        """Diagonal of the hat_matrix using only exog as in OLS

        """
        get_exogs = getattr(self.results.model, '_get_exogs', None)
        if get_exogs is not None:
            exog = np.column_stack(get_exogs())
        else:
            exog = self.exog
        return (exog * np.linalg.pinv(exog).T).sum(1)

    @cache_readonly
    def d_params(self):
        """Approximate change in parameter estimates when dropping observation.

        This uses one-step approximation of the parameter change to deleting
        one observation.
        """
        so_noti = self.score_obs.sum(0) - self.score_obs
        beta_i = np.linalg.solve(self.hessian, so_noti.T).T
        if self.hat_matrix_diag is not None:
            beta_i /= (1 - self.hat_matrix_diag)[:, None]
        return beta_i

    @cache_readonly
    def dfbetas(self):
        """Scaled change in parameter estimates.

        The one-step change of parameters in d_params is rescaled by dividing
        by the standard error of the parameter estimate given by results.bse.
        """
        beta_i = self.d_params / self.results.bse
        return beta_i

    @cache_readonly
    def params_one(self):
        """Parameter estimate based on one-step approximation.

        This the one step parameter estimate computed as
        ``params`` from the full sample minus ``d_params``.
        """
        return self.results.params - self.d_params

    @cache_readonly
    def cooks_distance(self):
        """Cook's distance and p-values.

        Based on one step approximation d_params and on results.cov_params
        Cook's distance divides by the number of explanatory variables.

        p-values are based on the F-distribution which are only approximate
        outside of linear Gaussian models.

        Warning: The definition of p-values might change if we switch to using
        chi-square distribution instead of F-distribution, or if we make it
        dependent on the fit keyword use_t.
        """
        cooks_d2 = (self.d_params * np.linalg.solve(self.cov_params, self.d_params.T).T).sum(1)
        cooks_d2 /= self.k_params
        from scipy import stats
        pvals = stats.f.sf(cooks_d2, self.k_params, self.results.df_resid)
        return (cooks_d2, pvals)

    @cache_readonly
    def resid_studentized(self):
        """studentized default residuals.

        This uses the residual in `resid` attribute, which is by default
        resid_pearson and studentizes is using the generalized leverage.

        self.resid / np.sqrt(1 - self.hat_matrix_diag)

        Studentized residuals are not available if hat_matrix_diag is None.

        """
        return self.resid / np.sqrt(1 - self.hat_matrix_diag)

    def resid_score_factor(self):
        """Score residual divided by sqrt of hessian factor.

        experimental, agrees with GLMInfluence for Binomial and Gaussian.
        This corresponds to considering the linear predictors as parameters
        of the model.

        Note: Nhis might have nan values if second derivative, hessian_factor,
        is positive, i.e. loglikelihood is not globally concave w.r.t. linear
        predictor. (This occured in an example for GeneralizedPoisson)
        """
        from statsmodels.genmod.generalized_linear_model import GLM
        sf = self.results.model.score_factor(self.results.params)
        hf = self.results.model.hessian_factor(self.results.params)
        if isinstance(sf, tuple):
            sf = sf[0]
        if isinstance(hf, tuple):
            hf = hf[0]
        if not isinstance(self.results.model, GLM):
            hf = -hf
        return sf / np.sqrt(hf) / np.sqrt(1 - self.hat_matrix_diag)

    def resid_score(self, joint=True, index=None, studentize=False):
        """Score observations scaled by inverse hessian.

        Score residual in resid_score are defined in analogy to a score test
        statistic for each observation.

        Parameters
        ----------
        joint : bool
            If joint is true, then a quadratic form similar to score_test is
            returned for each observation.
            If joint is false, then standardized score_obs are returned. The
            returned array is two-dimensional
        index : ndarray (optional)
            Optional index to select a subset of score_obs columns.
            By default, all columns of score_obs will be used.
        studentize : bool
            If studentize is true, the the scaled residuals are also
            studentized using the generalized leverage.

        Returns
        -------
        array :  1-D or 2-D residuals

        Notes
        -----
        Status: experimental

        Because of the one srep approacimation of d_params, score residuals
        are identical to cooks_distance, except for

        - cooks_distance is normalized by the number of parameters
        - cooks_distance uses cov_params, resid_score is based on Hessian.
          This will make them differ in the case of robust cov_params.

        """
        score_obs = self.results.model.score_obs(self.results.params)
        hess = self.results.model.hessian(self.results.params)
        if index is not None:
            score_obs = score_obs[:, index]
            hess = hess[index[:, None], index]
        if joint:
            resid = (score_obs.T * np.linalg.solve(-hess, score_obs.T)).sum(0)
        else:
            resid = score_obs / np.sqrt(np.diag(-hess))
        if studentize:
            if joint:
                resid /= np.sqrt(1 - self.hat_matrix_diag)
            else:
                resid /= np.sqrt(1 - self.hat_matrix_diag[:, None])
        return resid

    @cache_readonly
    def _get_prediction(self):
        with warnings.catch_warnings():
            msg = 'linear keyword is deprecated, use which="linear"'
            warnings.filterwarnings('ignore', message=msg, category=FutureWarning)
            pred = self.results.get_prediction()
        return pred

    @cache_readonly
    def d_fittedvalues(self):
        """Change in expected response, fittedvalues.

        Local change of expected mean given the change in the parameters as
        computed in d_params.

        Notes
        -----
        This uses the one-step approximation of the parameter change to
        deleting one observation ``d_params``.
        """
        params = np.asarray(self.results.params)
        deriv = self.results.model._deriv_mean_dparams(params)
        return (deriv * self.d_params).sum(1)

    @property
    def d_fittedvalues_scaled(self):
        """
        Change in fittedvalues scaled by standard errors.

        This uses one-step approximation of the parameter change to deleting
        one observation ``d_params``, and divides by the standard errors
        for the predicted mean provided by results.get_prediction.
        """
        return self.d_fittedvalues / self._get_prediction.se

    def summary_frame(self):
        """
        Creates a DataFrame with influence results.

        Returns
        -------
        frame : pandas DataFrame
            A DataFrame with selected results for each observation.
            The index will be the same as provided to the model.

        Notes
        -----
        The resultant DataFrame contains six variables in addition to the
        ``dfbetas``. These are:

        * cooks_d : Cook's Distance defined in ``cooks_distance``
        * standard_resid : Standardized residuals defined in
          `resid_studentizedl`
        * hat_diag : The diagonal of the projection, or hat, matrix defined in
          `hat_matrix_diag`. Not included if None.
        * dffits_internal : DFFITS statistics using internally Studentized
          residuals defined in `d_fittedvalues_scaled`
        """
        from pandas import DataFrame
        data = self.results.model.data
        row_labels = data.row_labels
        beta_labels = ['dfb_' + i for i in data.xnames]
        if self.hat_matrix_diag is not None:
            summary_data = DataFrame(dict(cooks_d=self.cooks_distance[0], standard_resid=self.resid_studentized, hat_diag=self.hat_matrix_diag, dffits_internal=self.d_fittedvalues_scaled), index=row_labels)
        else:
            summary_data = DataFrame(dict(cooks_d=self.cooks_distance[0], dffits_internal=self.d_fittedvalues_scaled), index=row_labels)
        dfbeta = DataFrame(self.dfbetas, columns=beta_labels, index=row_labels)
        return dfbeta.join(summary_data)
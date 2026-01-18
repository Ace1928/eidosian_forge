from collections.abc import Iterable
import copy  # check if needed when dropping python 2.7
import numpy as np
from scipy import optimize
import pandas as pd
import statsmodels.base.wrapper as wrap
from statsmodels.discrete.discrete_model import Logit
from statsmodels.genmod.generalized_linear_model import (
import statsmodels.regression.linear_model as lm
from statsmodels.tools.sm_exceptions import (PerfectSeparationError,
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tools.linalg import matrix_sqrt
from statsmodels.base._penalized import PenalizedMixin
from statsmodels.gam.gam_penalties import MultivariateGamPenalty
from statsmodels.gam.gam_cross_validation.gam_cross_validation import (
from statsmodels.gam.gam_cross_validation.cross_validators import KFold
class GLMGamResults(GLMResults):
    """Results class for generalized additive models, GAM.

    This inherits from GLMResults.

    Warning: some inherited methods might not correctly take account of the
    penalization

    GLMGamResults inherits from GLMResults
    All methods related to the loglikelihood function return the penalized
    values.

    Attributes
    ----------

    edf
        list of effective degrees of freedom for each column of the design
        matrix.
    hat_matrix_diag
        diagonal of hat matrix
    gcv
        generalized cross-validation criterion computed as
        ``gcv = scale / (1. - hat_matrix_trace / self.nobs)**2``
    cv
        cross-validation criterion computed as
        ``cv = ((resid_pearson / (1 - hat_matrix_diag))**2).sum() / nobs``

    Notes
    -----
    status: experimental
    """

    def __init__(self, model, params, normalized_cov_params, scale, **kwds):
        self.model = model
        self.params = params
        self.normalized_cov_params = normalized_cov_params
        self.scale = scale
        edf = self.edf.sum()
        self.df_model = edf - 1
        self.df_resid = self.model.endog.shape[0] - edf
        self.model.df_model = self.df_model
        self.model.df_resid = self.df_resid
        mu = self.fittedvalues
        self.scale = scale = self.model.estimate_scale(mu)
        super().__init__(model, params, normalized_cov_params, scale, **kwds)

    def _tranform_predict_exog(self, exog=None, exog_smooth=None, transform=True):
        """Transform original explanatory variables for prediction

        Parameters
        ----------
        exog : array_like, optional
            The values for the linear explanatory variables.
        exog_smooth : array_like
            values for the variables in the smooth terms
        transform : bool, optional
            If transform is False, then ``exog`` is returned unchanged and
            ``x`` is ignored. It is assumed that exog contains the full
            design matrix for the predict observations.
            If transform is True, then the basis representation of the smooth
            term will be constructed from the provided ``x``.

        Returns
        -------
        exog_transformed : ndarray
            design matrix for the prediction
        """
        if exog_smooth is not None:
            exog_smooth = np.asarray(exog_smooth)
        exog_index = None
        if transform is False:
            if exog_smooth is None:
                ex = exog
            elif exog is None:
                ex = exog_smooth
            else:
                ex = np.column_stack((exog, exog_smooth))
        else:
            if exog is not None and hasattr(self.model, 'design_info_linear'):
                exog, exog_index = _transform_predict_exog(self.model, exog, self.model.design_info_linear)
            if exog_smooth is not None:
                ex_smooth = self.model.smoother.transform(exog_smooth)
                if exog is None:
                    ex = ex_smooth
                else:
                    ex = np.column_stack((exog, ex_smooth))
            else:
                ex = exog
        return (ex, exog_index)

    def predict(self, exog=None, exog_smooth=None, transform=True, **kwargs):
        """"
        compute prediction

        Parameters
        ----------
        exog : array_like, optional
            The values for the linear explanatory variables
        exog_smooth : array_like
            values for the variables in the smooth terms
        transform : bool, optional
            If transform is True, then the basis representation of the smooth
            term will be constructed from the provided ``exog``.
        kwargs :
            Some models can take additional arguments or keywords, see the
            predict method of the model for the details.

        Returns
        -------
        prediction : ndarray, pandas.Series or pandas.DataFrame
            predicted values
        """
        ex, exog_index = self._tranform_predict_exog(exog=exog, exog_smooth=exog_smooth, transform=transform)
        predict_results = super().predict(ex, transform=False, **kwargs)
        if exog_index is not None and (not hasattr(predict_results, 'predicted_values')):
            if predict_results.ndim == 1:
                return pd.Series(predict_results, index=exog_index)
            else:
                return pd.DataFrame(predict_results, index=exog_index)
        else:
            return predict_results

    def get_prediction(self, exog=None, exog_smooth=None, transform=True, **kwargs):
        """compute prediction results

        Parameters
        ----------
        exog : array_like, optional
            The values for which you want to predict.
        exog_smooth : array_like
            values for the variables in the smooth terms
        transform : bool, optional
            If transform is True, then the basis representation of the smooth
            term will be constructed from the provided ``x``.
        kwargs :
            Some models can take additional arguments or keywords, see the
            predict method of the model for the details.

        Returns
        -------
        prediction_results : generalized_linear_model.PredictionResults
            The prediction results instance contains prediction and prediction
            variance and can on demand calculate confidence intervals and
            summary tables for the prediction of the mean and of new
            observations.
        """
        ex, exog_index = self._tranform_predict_exog(exog=exog, exog_smooth=exog_smooth, transform=transform)
        return super().get_prediction(ex, transform=False, **kwargs)

    def partial_values(self, smooth_index, include_constant=True):
        """contribution of a smooth term to the linear prediction

        Warning: This will be replaced by a predict method

        Parameters
        ----------
        smooth_index : int
            index of the smooth term within list of smooth terms
        include_constant : bool
            If true, then the estimated intercept is added to the prediction
            and its standard errors. This avoids that the confidence interval
            has zero width at the imposed identification constraint, e.g.
            either at a reference point or at the mean.

        Returns
        -------
        predicted : nd_array
            predicted value of linear term.
            This is not the expected response if the link function is not
            linear.
        se_pred : nd_array
            standard error of linear prediction
        """
        variable = smooth_index
        smoother = self.model.smoother
        mask = smoother.mask[variable]
        start_idx = self.model.k_exog_linear
        idx = start_idx + np.nonzero(mask)[0]
        exog_part = smoother.basis[:, mask]
        const_idx = self.model.data.const_idx
        if include_constant and const_idx is not None:
            idx = np.concatenate(([const_idx], idx))
            exog_part = self.model.exog[:, idx]
        linpred = np.dot(exog_part, self.params[idx])
        partial_cov_params = self.cov_params(column=idx)
        covb = partial_cov_params
        var = (exog_part * np.dot(covb, exog_part.T).T).sum(1)
        se = np.sqrt(var)
        return (linpred, se)

    def plot_partial(self, smooth_index, plot_se=True, cpr=False, include_constant=True, ax=None):
        """plot the contribution of a smooth term to the linear prediction

        Parameters
        ----------
        smooth_index : int
            index of the smooth term within list of smooth terms
        plot_se : bool
            If plot_se is true, then the confidence interval for the linear
            prediction will be added to the plot.
        cpr : bool
            If cpr (component plus residual) is true, then a scatter plot of
            the partial working residuals will be added to the plot.
        include_constant : bool
            If true, then the estimated intercept is added to the prediction
            and its standard errors. This avoids that the confidence interval
            has zero width at the imposed identification constraint, e.g.
            either at a reference point or at the mean.
        ax : None or matplotlib axis instance
           If ax is not None, then the plot will be added to it.

        Returns
        -------
        Figure
            If `ax` is None, the created figure. Otherwise, the Figure to which
            `ax` is connected.
        """
        from statsmodels.graphics.utils import _import_mpl, create_mpl_ax
        _import_mpl()
        variable = smooth_index
        y_est, se = self.partial_values(variable, include_constant=include_constant)
        smoother = self.model.smoother
        x = smoother.smoothers[variable].x
        sort_index = np.argsort(x)
        x = x[sort_index]
        y_est = y_est[sort_index]
        se = se[sort_index]
        fig, ax = create_mpl_ax(ax)
        if cpr:
            residual = self.resid_working[sort_index]
            cpr_ = y_est + residual
            ax.scatter(x, cpr_, s=4)
        ax.plot(x, y_est, c='blue', lw=2)
        if plot_se:
            ax.plot(x, y_est + 1.96 * se, '-', c='blue')
            ax.plot(x, y_est - 1.96 * se, '-', c='blue')
        ax.set_xlabel(smoother.smoothers[variable].variable_name)
        return fig

    def test_significance(self, smooth_index):
        """hypothesis test that a smooth component is zero.

        This calls `wald_test` to compute the hypothesis test, but uses
        effective degrees of freedom.

        Parameters
        ----------
        smooth_index : int
            index of the smooth term within list of smooth terms

        Returns
        -------
        wald_test : ContrastResults instance
            the results instance created by `wald_test`
        """
        variable = smooth_index
        smoother = self.model.smoother
        start_idx = self.model.k_exog_linear
        k_params = len(self.params)
        mask = smoother.mask[variable]
        k_constraints = mask.sum()
        idx = start_idx + np.nonzero(mask)[0][0]
        constraints = np.eye(k_constraints, k_params, idx)
        df_constraints = self.edf[idx:idx + k_constraints].sum()
        return self.wald_test(constraints, df_constraints=df_constraints)

    def get_hat_matrix_diag(self, observed=True, _axis=1):
        """
        Compute the diagonal of the hat matrix

        Parameters
        ----------
        observed : bool
            If true, then observed hessian is used in the hat matrix
            computation. If false, then the expected hessian is used.
            In the case of a canonical link function both are the same.
            This is only relevant for models that implement both observed
            and expected Hessian, which is currently only GLM. Other
            models only use the observed Hessian.
        _axis : int
            This is mainly for internal use. By default it returns the usual
            diagonal of the hat matrix. If _axis is zero, then the result
            corresponds to the effective degrees of freedom, ``edf`` for each
            column of exog.

        Returns
        -------
        hat_matrix_diag : ndarray
            The diagonal of the hat matrix computed from the observed
            or expected hessian.
        """
        weights = self.model.hessian_factor(self.params, scale=self.scale, observed=observed)
        wexog = np.sqrt(weights)[:, None] * self.model.exog
        hess_inv = self.normalized_cov_params * self.scale
        hd = (wexog * hess_inv.dot(wexog.T).T).sum(axis=_axis)
        return hd

    @cache_readonly
    def edf(self):
        return self.get_hat_matrix_diag(_axis=0)

    @cache_readonly
    def hat_matrix_trace(self):
        return self.hat_matrix_diag.sum()

    @cache_readonly
    def hat_matrix_diag(self):
        return self.get_hat_matrix_diag(observed=True)

    @cache_readonly
    def gcv(self):
        return self.scale / (1.0 - self.hat_matrix_trace / self.nobs) ** 2

    @cache_readonly
    def cv(self):
        cv_ = ((self.resid_pearson / (1.0 - self.hat_matrix_diag)) ** 2).sum()
        cv_ /= self.nobs
        return cv_
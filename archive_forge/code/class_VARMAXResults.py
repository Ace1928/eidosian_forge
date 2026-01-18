import contextlib
from warnings import warn
import pandas as pd
import numpy as np
from statsmodels.compat.pandas import Appender
from statsmodels.tools.tools import Bunch
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tsa.vector_ar import var_model
import statsmodels.base.wrapper as wrap
from statsmodels.tools.sm_exceptions import EstimationWarning
from .kalman_filter import INVERT_UNIVARIATE, SOLVE_LU
from .mlemodel import MLEModel, MLEResults, MLEResultsWrapper
from .initialization import Initialization
from .tools import (
class VARMAXResults(MLEResults):
    """
    Class to hold results from fitting an VARMAX model.

    Parameters
    ----------
    model : VARMAX instance
        The fitted model instance

    Attributes
    ----------
    specification : dictionary
        Dictionary including all attributes from the VARMAX model instance.
    coefficient_matrices_var : ndarray
        Array containing autoregressive lag polynomial coefficient matrices,
        ordered from lowest degree to highest.
    coefficient_matrices_vma : ndarray
        Array containing moving average lag polynomial coefficients,
        ordered from lowest degree to highest.

    See Also
    --------
    statsmodels.tsa.statespace.kalman_filter.FilterResults
    statsmodels.tsa.statespace.mlemodel.MLEResults
    """

    def __init__(self, model, params, filter_results, cov_type=None, cov_kwds=None, **kwargs):
        super().__init__(model, params, filter_results, cov_type, cov_kwds, **kwargs)
        self.specification = Bunch(**{'error_cov_type': self.model.error_cov_type, 'measurement_error': self.model.measurement_error, 'enforce_stationarity': self.model.enforce_stationarity, 'enforce_invertibility': self.model.enforce_invertibility, 'trend_offset': self.model.trend_offset, 'order': self.model.order, 'k_ar': self.model.k_ar, 'k_ma': self.model.k_ma, 'trend': self.model.trend, 'k_trend': self.model.k_trend, 'k_exog': self.model.k_exog})
        self.coefficient_matrices_var = None
        self.coefficient_matrices_vma = None
        if self.model.k_ar > 0:
            ar_params = np.array(self.params[self.model._params_ar])
            k_endog = self.model.k_endog
            k_ar = self.model.k_ar
            self.coefficient_matrices_var = ar_params.reshape(k_endog * k_ar, k_endog).T.reshape(k_endog, k_endog, k_ar).T
        if self.model.k_ma > 0:
            ma_params = np.array(self.params[self.model._params_ma])
            k_endog = self.model.k_endog
            k_ma = self.model.k_ma
            self.coefficient_matrices_vma = ma_params.reshape(k_endog * k_ma, k_endog).T.reshape(k_endog, k_endog, k_ma).T

    def extend(self, endog, exog=None, **kwargs):
        if exog is not None:
            fcast = self.get_prediction(self.nobs, self.nobs, exog=exog[:1])
            fcast_results = fcast.prediction_results
            initial_state = fcast_results.predicted_state[..., 0]
            initial_state_cov = fcast_results.predicted_state_cov[..., 0]
        else:
            initial_state = self.predicted_state[..., -1]
            initial_state_cov = self.predicted_state_cov[..., -1]
        kwargs.setdefault('trend_offset', self.nobs + self.model.trend_offset)
        mod = self.model.clone(endog, exog=exog, **kwargs)
        mod.ssm.initialization = Initialization(mod.k_states, 'known', constant=initial_state, stationary_cov=initial_state_cov)
        if self.smoother_results is not None:
            res = mod.smooth(self.params)
        else:
            res = mod.filter(self.params)
        return res

    @contextlib.contextmanager
    def _set_final_exog(self, exog):
        """
        Set the final state intercept value using out-of-sample `exog` / trend

        Parameters
        ----------
        exog : ndarray
            Out-of-sample `exog` values, usually produced by
            `_validate_out_of_sample_exog` to ensure the correct shape (this
            method does not do any additional validation of its own).
        out_of_sample : int
            Number of out-of-sample periods.

        Notes
        -----
        This context manager calls the model-level context manager and
        additionally updates the last element of filter_results.state_intercept
        appropriately.
        """
        mod = self.model
        with mod._set_final_exog(exog):
            cache_value = self.filter_results.state_intercept[:, -1]
            mod.update(self.params)
            self.filter_results.state_intercept[:mod.k_endog, -1] = mod['state_intercept', :mod.k_endog, -1]
            try:
                yield
            finally:
                self.filter_results.state_intercept[:, -1] = cache_value

    @contextlib.contextmanager
    def _set_final_predicted_state(self, exog, out_of_sample):
        """
        Set the final predicted state value using out-of-sample `exog` / trend

        Parameters
        ----------
        exog : ndarray
            Out-of-sample `exog` values, usually produced by
            `_validate_out_of_sample_exog` to ensure the correct shape (this
            method does not do any additional validation of its own).
        out_of_sample : int
            Number of out-of-sample periods.

        Notes
        -----
        We need special handling for forecasting with `exog`, because
        if we had these then the last predicted_state has been set to NaN since
        we did not have the appropriate `exog` to create it.
        """
        flag = out_of_sample and self.model.k_exog > 0
        if flag:
            tmp_endog = concat([self.model.endog[-1:], np.zeros((1, self.model.k_endog))])
            if self.model.k_exog > 0:
                tmp_exog = concat([self.model.exog[-1:], exog[:1]])
            else:
                tmp_exog = None
            tmp_trend_offset = self.model.trend_offset + self.nobs - 1
            tmp_mod = self.model.clone(tmp_endog, exog=tmp_exog, trend_offset=tmp_trend_offset)
            constant = self.filter_results.predicted_state[:, -2]
            stationary_cov = self.filter_results.predicted_state_cov[:, :, -2]
            tmp_mod.ssm.initialize_known(constant=constant, stationary_cov=stationary_cov)
            tmp_res = tmp_mod.filter(self.params, transformed=True, includes_fixed=True, return_ssm=True)
            self.filter_results.predicted_state[:, -1] = tmp_res.predicted_state[:, -2]
        try:
            yield
        finally:
            if flag:
                self.filter_results.predicted_state[:, -1] = np.nan

    @Appender(MLEResults.get_prediction.__doc__)
    def get_prediction(self, start=None, end=None, dynamic=False, information_set='predicted', index=None, exog=None, **kwargs):
        if start is None:
            start = 0
        _start, _end, out_of_sample, _ = self.model._get_prediction_index(start, end, index, silent=True)
        exog = self.model._validate_out_of_sample_exog(exog, out_of_sample)
        extend_kwargs = {}
        if self.model.k_trend > 0:
            extend_kwargs['trend_offset'] = self.model.trend_offset + self.nobs
        with self._set_final_exog(exog):
            with self._set_final_predicted_state(exog, out_of_sample):
                out = super().get_prediction(start=start, end=end, dynamic=dynamic, information_set=information_set, index=index, exog=exog, extend_kwargs=extend_kwargs, **kwargs)
        return out

    @Appender(MLEResults.simulate.__doc__)
    def simulate(self, nsimulations, measurement_shocks=None, state_shocks=None, initial_state=None, anchor=None, repetitions=None, exog=None, extend_model=None, extend_kwargs=None, **kwargs):
        if anchor is None or anchor == 'start':
            iloc = 0
        elif anchor == 'end':
            iloc = self.nobs
        else:
            iloc, _, _ = self.model._get_index_loc(anchor)
        if iloc < 0:
            iloc = self.nobs + iloc
        if iloc > self.nobs:
            raise ValueError('Cannot anchor simulation after the estimated sample.')
        out_of_sample = max(iloc + nsimulations - self.nobs, 0)
        exog = self.model._validate_out_of_sample_exog(exog, out_of_sample)
        with self._set_final_predicted_state(exog, out_of_sample):
            out = super().simulate(nsimulations, measurement_shocks=measurement_shocks, state_shocks=state_shocks, initial_state=initial_state, anchor=anchor, repetitions=repetitions, exog=exog, extend_model=extend_model, extend_kwargs=extend_kwargs, **kwargs)
        return out

    def _news_previous_results(self, previous, start, end, periods, revisions_details_start=False, state_index=None):
        exog = None
        out_of_sample = self.nobs - previous.nobs
        if self.model.k_exog > 0 and out_of_sample > 0:
            exog = self.model.exog[-out_of_sample:]
        with contextlib.ExitStack() as stack:
            stack.enter_context(previous.model._set_final_exog(exog))
            stack.enter_context(previous._set_final_predicted_state(exog, out_of_sample))
            out = self.smoother_results.news(previous.smoother_results, start=start, end=end, revisions_details_start=revisions_details_start, state_index=state_index)
        return out

    @Appender(MLEResults.summary.__doc__)
    def summary(self, alpha=0.05, start=None, separate_params=True):
        from statsmodels.iolib.summary import summary_params
        spec = self.specification
        if spec.k_ar > 0 and spec.k_ma > 0:
            model_name = 'VARMA'
            order = '({},{})'.format(spec.k_ar, spec.k_ma)
        elif spec.k_ar > 0:
            model_name = 'VAR'
            order = '(%s)' % spec.k_ar
        else:
            model_name = 'VMA'
            order = '(%s)' % spec.k_ma
        if spec.k_exog > 0:
            model_name += 'X'
        model_name = [model_name + order]
        if spec.k_trend > 0:
            model_name.append('intercept')
        if spec.measurement_error:
            model_name.append('measurement error')
        summary = super().summary(alpha=alpha, start=start, model_name=model_name, display_params=not separate_params)
        if separate_params:
            indices = np.arange(len(self.params))

            def make_table(self, mask, title, strip_end=True):
                res = (self, self.params[mask], self.bse[mask], self.zvalues[mask], self.pvalues[mask], self.conf_int(alpha)[mask])
                param_names = []
                for name in np.array(self.data.param_names)[mask].tolist():
                    if strip_end:
                        param_name = '.'.join(name.split('.')[:-1])
                    else:
                        param_name = name
                    if name in self.fixed_params:
                        param_name = '%s (fixed)' % param_name
                    param_names.append(param_name)
                return summary_params(res, yname=None, xname=param_names, alpha=alpha, use_t=False, title=title)
            k_endog = self.model.k_endog
            k_ar = self.model.k_ar
            k_ma = self.model.k_ma
            k_trend = self.model.k_trend
            k_exog = self.model.k_exog
            endog_masks = []
            for i in range(k_endog):
                masks = []
                offset = 0
                if k_trend > 0:
                    masks.append(np.arange(i, i + k_endog * k_trend, k_endog))
                    offset += k_endog * k_trend
                if k_ar > 0:
                    start = i * k_endog * k_ar
                    end = (i + 1) * k_endog * k_ar
                    masks.append(offset + np.arange(start, end))
                    offset += k_ar * k_endog ** 2
                if k_ma > 0:
                    start = i * k_endog * k_ma
                    end = (i + 1) * k_endog * k_ma
                    masks.append(offset + np.arange(start, end))
                    offset += k_ma * k_endog ** 2
                if k_exog > 0:
                    masks.append(offset + np.arange(i * k_exog, (i + 1) * k_exog))
                    offset += k_endog * k_exog
                if self.model.measurement_error:
                    masks.append(np.array(self.model.k_params - i - 1, ndmin=1))
                mask = np.concatenate(masks)
                endog_masks.append(mask)
                endog_names = self.model.endog_names
                if not isinstance(endog_names, list):
                    endog_names = [endog_names]
                title = 'Results for equation %s' % endog_names[i]
                table = make_table(self, mask, title)
                summary.tables.append(table)
            state_cov_mask = np.arange(len(self.params))[self.model._params_state_cov]
            table = make_table(self, state_cov_mask, 'Error covariance matrix', strip_end=False)
            summary.tables.append(table)
            masks = []
            for m in (endog_masks, [state_cov_mask]):
                m = np.array(m).flatten()
                if len(m) > 0:
                    masks.append(m)
            masks = np.concatenate(masks)
            inverse_mask = np.array(list(set(indices).difference(set(masks))))
            if len(inverse_mask) > 0:
                table = make_table(self, inverse_mask, 'Other parameters', strip_end=False)
                summary.tables.append(table)
        return summary
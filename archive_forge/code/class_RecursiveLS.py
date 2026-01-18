import numpy as np
import pandas as pd
from statsmodels.compat.pandas import Appender
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tsa.statespace.mlemodel import (
from statsmodels.tsa.statespace.tools import concat
from statsmodels.tools.tools import Bunch
from statsmodels.tools.decorators import cache_readonly
import statsmodels.base.wrapper as wrap
class RecursiveLS(MLEModel):
    """
    Recursive least squares

    Parameters
    ----------
    endog : array_like
        The observed time-series process :math:`y`
    exog : array_like
        Array of exogenous regressors, shaped nobs x k.
    constraints : array_like, str, or tuple
            - array : An r x k array where r is the number of restrictions to
              test and k is the number of regressors. It is assumed that the
              linear combination is equal to zero.
            - str : The full hypotheses to test can be given as a string.
              See the examples.
            - tuple : A tuple of arrays in the form (R, q), ``q`` can be
              either a scalar or a length p row vector.

    Notes
    -----
    Recursive least squares (RLS) corresponds to expanding window ordinary
    least squares (OLS).

    This model applies the Kalman filter to compute recursive estimates of the
    coefficients and recursive residuals.

    References
    ----------
    .. [*] Durbin, James, and Siem Jan Koopman. 2012.
       Time Series Analysis by State Space Methods: Second Edition.
       Oxford University Press.
    """

    def __init__(self, endog, exog, constraints=None, **kwargs):
        endog_using_pandas = _is_using_pandas(endog, None)
        if not endog_using_pandas:
            endog = np.asanyarray(endog)
        exog_is_using_pandas = _is_using_pandas(exog, None)
        if not exog_is_using_pandas:
            exog = np.asarray(exog)
        if exog.ndim == 1:
            if not exog_is_using_pandas:
                exog = exog[:, None]
            else:
                exog = pd.DataFrame(exog)
        self.k_exog = exog.shape[1]
        self.k_constraints = 0
        self._r_matrix = self._q_matrix = None
        if constraints is not None:
            from patsy import DesignInfo
            from statsmodels.base.data import handle_data
            data = handle_data(endog, exog, **kwargs)
            names = data.param_names
            LC = DesignInfo(names).linear_constraint(constraints)
            self._r_matrix, self._q_matrix = (LC.coefs, LC.constants)
            self.k_constraints = self._r_matrix.shape[0]
            nobs = len(endog)
            constraint_endog = np.zeros((nobs, len(self._r_matrix)))
            if endog_using_pandas:
                constraint_endog = pd.DataFrame(constraint_endog, index=endog.index)
                endog = concat([endog, constraint_endog], axis=1)
                endog.iloc[:, 1:] = np.tile(self._q_matrix.T, (nobs, 1))
            else:
                endog[:, 1:] = self._q_matrix[:, 0]
        kwargs.setdefault('initialization', 'diffuse')
        formula_kwargs = ['missing', 'missing_idx', 'formula', 'design_info']
        for name in formula_kwargs:
            if name in kwargs:
                del kwargs[name]
        super().__init__(endog, k_states=self.k_exog, exog=exog, **kwargs)
        self.ssm.filter_univariate = True
        self.ssm.filter_concentrated = True
        self['design'] = np.zeros((self.k_endog, self.k_states, self.nobs))
        self['design', 0] = self.exog[:, :, None].T
        if self._r_matrix is not None:
            self['design', 1:, :] = self._r_matrix[:, :, None]
        self['transition'] = np.eye(self.k_states)
        self['obs_cov', 0, 0] = 1.0
        self['transition'] = np.eye(self.k_states)
        if self._r_matrix is not None:
            self.k_endog = 1

    @classmethod
    def from_formula(cls, formula, data, subset=None, constraints=None):
        return super(MLEModel, cls).from_formula(formula, data, subset, constraints=constraints)

    def _validate_can_fix_params(self, param_names):
        raise ValueError('Linear constraints on coefficients should be given using the `constraints` argument in constructing. the model. Other parameter constraints are not available in the resursive least squares model.')

    def fit(self):
        """
        Fits the model by application of the Kalman filter

        Returns
        -------
        RecursiveLSResults
        """
        smoother_results = self.smooth(return_ssm=True)
        with self.ssm.fixed_scale(smoother_results.scale):
            res = self.smooth()
        return res

    def filter(self, return_ssm=False, **kwargs):
        result = super().filter([], transformed=True, cov_type='none', return_ssm=True, **kwargs)
        if not return_ssm:
            params = result.filtered_state[:, -1]
            cov_kwds = {'custom_cov_type': 'nonrobust', 'custom_cov_params': result.filtered_state_cov[:, :, -1], 'custom_description': 'Parameters and covariance matrix estimates are RLS estimates conditional on the entire sample.'}
            result = RecursiveLSResultsWrapper(RecursiveLSResults(self, params, result, cov_type='custom', cov_kwds=cov_kwds))
        return result

    def smooth(self, return_ssm=False, **kwargs):
        result = super().smooth([], transformed=True, cov_type='none', return_ssm=True, **kwargs)
        if not return_ssm:
            params = result.filtered_state[:, -1]
            cov_kwds = {'custom_cov_type': 'nonrobust', 'custom_cov_params': result.filtered_state_cov[:, :, -1], 'custom_description': 'Parameters and covariance matrix estimates are RLS estimates conditional on the entire sample.'}
            result = RecursiveLSResultsWrapper(RecursiveLSResults(self, params, result, cov_type='custom', cov_kwds=cov_kwds))
        return result

    @property
    def endog_names(self):
        endog_names = super().endog_names
        return endog_names[0] if isinstance(endog_names, list) else endog_names

    @property
    def param_names(self):
        return self.exog_names

    @property
    def start_params(self):
        return np.zeros(0)

    def update(self, params, **kwargs):
        """
        Update the parameters of the model

        Updates the representation matrices to fill in the new parameter
        values.

        Parameters
        ----------
        params : array_like
            Array of new parameters.
        transformed : bool, optional
            Whether or not `params` is already transformed. If set to False,
            `transform_params` is called. Default is True..

        Returns
        -------
        params : array_like
            Array of parameters.
        """
        pass
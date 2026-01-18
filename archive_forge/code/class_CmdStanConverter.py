import logging
import os
import re
from collections import defaultdict
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
from .. import utils
from ..rcparams import rcParams
from .base import CoordSpec, DimSpec, dict_to_dataset, infer_stan_dtypes, requires
from .inference_data import InferenceData
class CmdStanConverter:
    """Encapsulate CmdStan specific logic."""

    def __init__(self, *, posterior=None, posterior_predictive=None, predictions=None, prior=None, prior_predictive=None, observed_data=None, observed_data_var=None, constant_data=None, constant_data_var=None, predictions_constant_data=None, predictions_constant_data_var=None, log_likelihood=None, index_origin=None, coords=None, dims=None, disable_glob=False, save_warmup=None, dtypes=None):
        self.posterior_ = check_glob(posterior, 'posterior', disable_glob)
        self.posterior_predictive = check_glob(posterior_predictive, 'posterior_predictive', disable_glob)
        self.predictions = check_glob(predictions, 'predictions', disable_glob)
        self.prior_ = check_glob(prior, 'prior', disable_glob)
        self.prior_predictive = check_glob(prior_predictive, 'prior_predictive', disable_glob)
        self.log_likelihood = check_glob(log_likelihood, 'log_likelihood', disable_glob)
        self.observed_data = observed_data
        self.observed_data_var = observed_data_var
        self.constant_data = constant_data
        self.constant_data_var = constant_data_var
        self.predictions_constant_data = predictions_constant_data
        self.predictions_constant_data_var = predictions_constant_data_var
        self.coords = coords if coords is not None else {}
        self.dims = dims if dims is not None else {}
        self.posterior = None
        self.prior = None
        self.attrs = None
        self.attrs_prior = None
        self.save_warmup = rcParams['data.save_warmup'] if save_warmup is None else save_warmup
        self.index_origin = index_origin
        if dtypes is None:
            dtypes = {}
        elif isinstance(dtypes, str):
            dtypes_path = Path(dtypes)
            if dtypes_path.exists():
                with dtypes_path.open('r', encoding='UTF-8') as f_obj:
                    model_code = f_obj.read()
            else:
                model_code = dtypes
            dtypes = infer_stan_dtypes(model_code)
        self.dtypes = dtypes
        self._parse_posterior()
        self._parse_prior()
        if self.log_likelihood is None and self.posterior_ is not None and any((name.split('.')[0] == 'log_lik' for name in self.posterior_columns)):
            self.log_likelihood = ['log_lik']
        elif isinstance(self.log_likelihood, bool):
            self.log_likelihood = None

    @requires('posterior_')
    def _parse_posterior(self):
        """Read csv paths to list of ndarrays."""
        paths = self.posterior_
        if isinstance(paths, str):
            paths = [paths]
        chain_data = []
        columns = None
        for path in paths:
            output_data = _read_output(path)
            chain_data.append(output_data)
            if columns is None:
                columns = output_data
        self.posterior = ([item['sample'] for item in chain_data], [item['sample_warmup'] for item in chain_data])
        self.posterior_columns = columns['sample_columns']
        self.sample_stats_columns = columns['sample_stats_columns']
        attrs = {}
        for item in chain_data:
            for key, value in item['configuration_info'].items():
                if key not in attrs:
                    attrs[key] = []
                attrs[key].append(value)
        self.attrs = attrs

    @requires('prior_')
    def _parse_prior(self):
        """Read csv paths to list of ndarrays."""
        paths = self.prior_
        if isinstance(paths, str):
            paths = [paths]
        chain_data = []
        columns = None
        for path in paths:
            output_data = _read_output(path)
            chain_data.append(output_data)
            if columns is None:
                columns = output_data
        self.prior = ([item['sample'] for item in chain_data], [item['sample_warmup'] for item in chain_data])
        self.prior_columns = columns['sample_columns']
        self.sample_stats_prior_columns = columns['sample_stats_columns']
        attrs = {}
        for item in chain_data:
            for key, value in item['configuration_info'].items():
                if key not in attrs:
                    attrs[key] = []
                attrs[key].append(value)
        self.attrs_prior = attrs

    @requires('posterior')
    def posterior_to_xarray(self):
        """Extract posterior samples from output csv."""
        columns = self.posterior_columns
        posterior_predictive = self.posterior_predictive
        if posterior_predictive is None or (isinstance(posterior_predictive, str) and posterior_predictive.lower().endswith('.csv')):
            posterior_predictive = []
        elif isinstance(posterior_predictive, str):
            posterior_predictive = [col for col in columns if posterior_predictive == col.split('.')[0]]
        else:
            posterior_predictive = [col for col in columns if any((item == col.split('.')[0] for item in posterior_predictive))]
        predictions = self.predictions
        if predictions is None or (isinstance(predictions, str) and predictions.lower().endswith('.csv')):
            predictions = []
        elif isinstance(predictions, str):
            predictions = [col for col in columns if predictions == col.split('.')[0]]
        else:
            predictions = [col for col in columns if any((item == col.split('.')[0] for item in predictions))]
        log_likelihood = self.log_likelihood
        if log_likelihood is None or (isinstance(log_likelihood, str) and log_likelihood.lower().endswith('.csv')):
            log_likelihood = []
        elif isinstance(log_likelihood, str):
            log_likelihood = [col for col in columns if log_likelihood == col.split('.')[0]]
        elif isinstance(log_likelihood, dict):
            log_likelihood = [col for col in columns if any((item == col.split('.')[0] for item in log_likelihood.values()))]
        else:
            log_likelihood = [col for col in columns if any((item == col.split('.')[0] for item in log_likelihood))]
        invalid_cols = posterior_predictive + predictions + log_likelihood
        valid_cols = {col: idx for col, idx in columns.items() if col not in invalid_cols}
        data = _unpack_ndarrays(self.posterior[0], valid_cols, self.dtypes)
        data_warmup = _unpack_ndarrays(self.posterior[1], valid_cols, self.dtypes)
        return (dict_to_dataset(data, coords=self.coords, dims=self.dims, attrs=self.attrs, index_origin=self.index_origin), dict_to_dataset(data_warmup, coords=self.coords, dims=self.dims, attrs=self.attrs, index_origin=self.index_origin))

    @requires('posterior')
    @requires('sample_stats_columns')
    def sample_stats_to_xarray(self):
        """Extract sample_stats from fit."""
        dtypes = {'diverging': bool, 'n_steps': np.int64, 'tree_depth': np.int64, **self.dtypes}
        rename_dict = {'divergent': 'diverging', 'n_leapfrog': 'n_steps', 'treedepth': 'tree_depth', 'stepsize': 'step_size', 'accept_stat': 'acceptance_rate'}
        columns_new = {}
        for key, idx in self.sample_stats_columns.items():
            name = re.sub('__$', '', key)
            name = rename_dict.get(name, name)
            columns_new[name] = idx
        data = _unpack_ndarrays(self.posterior[0], columns_new, dtypes)
        data_warmup = _unpack_ndarrays(self.posterior[1], columns_new, dtypes)
        return (dict_to_dataset(data, coords=self.coords, dims=self.dims, attrs={item: key for key, item in rename_dict.items()}, index_origin=self.index_origin), dict_to_dataset(data_warmup, coords=self.coords, dims=self.dims, attrs={item: key for key, item in rename_dict.items()}, index_origin=self.index_origin))

    @requires('posterior')
    @requires('posterior_predictive')
    def posterior_predictive_to_xarray(self):
        """Convert posterior_predictive samples to xarray."""
        posterior_predictive = self.posterior_predictive
        if isinstance(posterior_predictive, (tuple, list)) and posterior_predictive[0].endswith('.csv') or (isinstance(posterior_predictive, str) and posterior_predictive.endswith('.csv')):
            if isinstance(posterior_predictive, str):
                posterior_predictive = [posterior_predictive]
            chain_data = []
            chain_data_warmup = []
            columns = None
            attrs = {}
            for path in posterior_predictive:
                parsed_output = _read_output(path)
                chain_data.append(parsed_output['sample'])
                chain_data_warmup.append(parsed_output['sample_warmup'])
                if columns is None:
                    columns = parsed_output['sample_columns']
                for key, value in parsed_output['configuration_info'].items():
                    if key not in attrs:
                        attrs[key] = []
                    attrs[key].append(value)
            data = _unpack_ndarrays(chain_data, columns, self.dtypes)
            data_warmup = _unpack_ndarrays(chain_data_warmup, columns, self.dtypes)
        else:
            if isinstance(posterior_predictive, str):
                posterior_predictive = [posterior_predictive]
            columns = {col: idx for col, idx in self.posterior_columns.items() if any((item == col.split('.')[0] for item in posterior_predictive))}
            data = _unpack_ndarrays(self.posterior[0], columns, self.dtypes)
            data_warmup = _unpack_ndarrays(self.posterior[1], columns, self.dtypes)
            attrs = None
        return (dict_to_dataset(data, coords=self.coords, dims=self.dims, attrs=attrs, index_origin=self.index_origin), dict_to_dataset(data_warmup, coords=self.coords, dims=self.dims, attrs=attrs, index_origin=self.index_origin))

    @requires('posterior')
    @requires('predictions')
    def predictions_to_xarray(self):
        """Convert out of sample predictions samples to xarray."""
        predictions = self.predictions
        if isinstance(predictions, (tuple, list)) and predictions[0].endswith('.csv') or (isinstance(predictions, str) and predictions.endswith('.csv')):
            if isinstance(predictions, str):
                predictions = [predictions]
            chain_data = []
            chain_data_warmup = []
            columns = None
            attrs = {}
            for path in predictions:
                parsed_output = _read_output(path)
                chain_data.append(parsed_output['sample'])
                chain_data_warmup.append(parsed_output['sample_warmup'])
                if columns is None:
                    columns = parsed_output['sample_columns']
                for key, value in parsed_output['configuration_info'].items():
                    if key not in attrs:
                        attrs[key] = []
                    attrs[key].append(value)
            data = _unpack_ndarrays(chain_data, columns, self.dtypes)
            data_warmup = _unpack_ndarrays(chain_data_warmup, columns, self.dtypes)
        else:
            if isinstance(predictions, str):
                predictions = [predictions]
            columns = {col: idx for col, idx in self.posterior_columns.items() if any((item == col.split('.')[0] for item in predictions))}
            data = _unpack_ndarrays(self.posterior[0], columns, self.dtypes)
            data_warmup = _unpack_ndarrays(self.posterior[1], columns, self.dtypes)
            attrs = None
        return (dict_to_dataset(data, coords=self.coords, dims=self.dims, attrs=attrs, index_origin=self.index_origin), dict_to_dataset(data_warmup, coords=self.coords, dims=self.dims, attrs=attrs, index_origin=self.index_origin))

    @requires('posterior')
    @requires('log_likelihood')
    def log_likelihood_to_xarray(self):
        """Convert elementwise log_likelihood samples to xarray."""
        log_likelihood = self.log_likelihood
        if isinstance(log_likelihood, (tuple, list)) and log_likelihood[0].endswith('.csv') or (isinstance(log_likelihood, str) and log_likelihood.endswith('.csv')):
            if isinstance(log_likelihood, str):
                log_likelihood = [log_likelihood]
            chain_data = []
            chain_data_warmup = []
            columns = None
            attrs = {}
            for path in log_likelihood:
                parsed_output = _read_output(path)
                chain_data.append(parsed_output['sample'])
                chain_data_warmup.append(parsed_output['sample_warmup'])
                if columns is None:
                    columns = parsed_output['sample_columns']
                for key, value in parsed_output['configuration_info'].items():
                    if key not in attrs:
                        attrs[key] = []
                    attrs[key].append(value)
            data = _unpack_ndarrays(chain_data, columns, self.dtypes)
            data_warmup = _unpack_ndarrays(chain_data_warmup, columns, self.dtypes)
        else:
            if isinstance(log_likelihood, dict):
                log_lik_to_obs_name = {v: k for k, v in log_likelihood.items()}
                columns = {col.replace(col_name, log_lik_to_obs_name[col_name]): idx for col, col_name, idx in ((col, col.split('.')[0], idx) for col, idx in self.posterior_columns.items()) if any((item == col_name for item in log_likelihood.values()))}
            else:
                if isinstance(log_likelihood, str):
                    log_likelihood = [log_likelihood]
                columns = {col: idx for col, idx in self.posterior_columns.items() if any((item == col.split('.')[0] for item in log_likelihood))}
            data = _unpack_ndarrays(self.posterior[0], columns, self.dtypes)
            data_warmup = _unpack_ndarrays(self.posterior[1], columns, self.dtypes)
            attrs = None
        return (dict_to_dataset(data, coords=self.coords, dims=self.dims, attrs=attrs, index_origin=self.index_origin, skip_event_dims=True), dict_to_dataset(data_warmup, coords=self.coords, dims=self.dims, attrs=attrs, index_origin=self.index_origin, skip_event_dims=True))

    @requires('prior')
    def prior_to_xarray(self):
        """Convert prior samples to xarray."""
        prior_predictive = self.prior_predictive
        columns = self.prior_columns
        if prior_predictive is None or (isinstance(prior_predictive, str) and prior_predictive.lower().endswith('.csv')):
            prior_predictive = []
        elif isinstance(prior_predictive, str):
            prior_predictive = [col for col in columns if prior_predictive == col.split('.')[0]]
        else:
            prior_predictive = [col for col in columns if any((item == col.split('.')[0] for item in prior_predictive))]
        invalid_cols = prior_predictive
        valid_cols = {col: idx for col, idx in columns.items() if col not in invalid_cols}
        data = _unpack_ndarrays(self.prior[0], valid_cols, self.dtypes)
        data_warmup = _unpack_ndarrays(self.prior[1], valid_cols, self.dtypes)
        return (dict_to_dataset(data, coords=self.coords, dims=self.dims, attrs=self.attrs_prior, index_origin=self.index_origin), dict_to_dataset(data_warmup, coords=self.coords, dims=self.dims, attrs=self.attrs_prior, index_origin=self.index_origin))

    @requires('prior')
    @requires('sample_stats_prior_columns')
    def sample_stats_prior_to_xarray(self):
        """Extract sample_stats from fit."""
        dtypes = {'diverging': bool, 'n_steps': np.int64, 'tree_depth': np.int64, **self.dtypes}
        rename_dict = {'divergent': 'diverging', 'n_leapfrog': 'n_steps', 'treedepth': 'tree_depth', 'stepsize': 'step_size', 'accept_stat': 'acceptance_rate'}
        columns_new = {}
        for key, idx in self.sample_stats_prior_columns.items():
            name = re.sub('__$', '', key)
            name = rename_dict.get(name, name)
            columns_new[name] = idx
        data = _unpack_ndarrays(self.posterior[0], columns_new, dtypes)
        data_warmup = _unpack_ndarrays(self.posterior[1], columns_new, dtypes)
        return (dict_to_dataset(data, coords=self.coords, dims=self.dims, attrs={item: key for key, item in rename_dict.items()}, index_origin=self.index_origin), dict_to_dataset(data_warmup, coords=self.coords, dims=self.dims, attrs={item: key for key, item in rename_dict.items()}, index_origin=self.index_origin))

    @requires('prior')
    @requires('prior_predictive')
    def prior_predictive_to_xarray(self):
        """Convert prior_predictive samples to xarray."""
        prior_predictive = self.prior_predictive
        if isinstance(prior_predictive, (tuple, list)) and prior_predictive[0].endswith('.csv') or (isinstance(prior_predictive, str) and prior_predictive.endswith('.csv')):
            if isinstance(prior_predictive, str):
                prior_predictive = [prior_predictive]
            chain_data = []
            chain_data_warmup = []
            columns = None
            attrs = {}
            for path in prior_predictive:
                parsed_output = _read_output(path)
                chain_data.append(parsed_output['sample'])
                chain_data_warmup.append(parsed_output['sample_warmup'])
                if columns is None:
                    columns = parsed_output['sample_columns']
                for key, value in parsed_output['configuration_info'].items():
                    if key not in attrs:
                        attrs[key] = []
                    attrs[key].append(value)
            data = _unpack_ndarrays(chain_data, columns, self.dtypes)
            data_warmup = _unpack_ndarrays(chain_data_warmup, columns, self.dtypes)
        else:
            if isinstance(prior_predictive, str):
                prior_predictive = [prior_predictive]
            columns = {col: idx for col, idx in self.prior_columns.items() if any((item == col.split('.')[0] for item in prior_predictive))}
            data = _unpack_ndarrays(self.prior[0], columns, self.dtypes)
            data_warmup = _unpack_ndarrays(self.prior[1], columns, self.dtypes)
            attrs = None
        return (dict_to_dataset(data, coords=self.coords, dims=self.dims, attrs=attrs, index_origin=self.index_origin), dict_to_dataset(data_warmup, coords=self.coords, dims=self.dims, attrs=attrs, index_origin=self.index_origin))

    @requires('observed_data')
    def observed_data_to_xarray(self):
        """Convert observed data to xarray."""
        observed_data_raw = _read_data(self.observed_data)
        variables = self.observed_data_var
        if isinstance(variables, str):
            variables = [variables]
        observed_data = {key: utils.one_de(vals) for key, vals in observed_data_raw.items() if variables is None or key in variables}
        return dict_to_dataset(observed_data, coords=self.coords, dims=self.dims, default_dims=[], index_origin=self.index_origin)

    @requires('constant_data')
    def constant_data_to_xarray(self):
        """Convert constant data to xarray."""
        constant_data_raw = _read_data(self.constant_data)
        variables = self.constant_data_var
        if isinstance(variables, str):
            variables = [variables]
        constant_data = {key: utils.one_de(vals) for key, vals in constant_data_raw.items() if variables is None or key in variables}
        return dict_to_dataset(constant_data, coords=self.coords, dims=self.dims, default_dims=[], index_origin=self.index_origin)

    @requires('predictions_constant_data')
    def predictions_constant_data_to_xarray(self):
        """Convert predictions constant data to xarray."""
        predictions_constant_data_raw = _read_data(self.predictions_constant_data)
        variables = self.predictions_constant_data_var
        if isinstance(variables, str):
            variables = [variables]
        predictions_constant_data = {}
        for key, vals in predictions_constant_data_raw.items():
            if variables is not None and key not in variables:
                continue
            vals = utils.one_de(vals)
            predictions_constant_data[key] = utils.one_de(vals)
        return dict_to_dataset(predictions_constant_data, coords=self.coords, dims=self.dims, default_dims=[], index_origin=self.index_origin)

    def to_inference_data(self):
        """Convert all available data to an InferenceData object.

        Note that if groups can not be created (i.e., there is no `output`, so
        the `posterior` and `sample_stats` can not be extracted), then the InferenceData
        will not have those groups.
        """
        return InferenceData(save_warmup=self.save_warmup, **{'posterior': self.posterior_to_xarray(), 'sample_stats': self.sample_stats_to_xarray(), 'log_likelihood': self.log_likelihood_to_xarray(), 'posterior_predictive': self.posterior_predictive_to_xarray(), 'prior': self.prior_to_xarray(), 'sample_stats_prior': self.sample_stats_prior_to_xarray(), 'prior_predictive': self.prior_predictive_to_xarray(), 'observed_data': self.observed_data_to_xarray(), 'constant_data': self.constant_data_to_xarray(), 'predictions': self.predictions_to_xarray(), 'predictions_constant_data': self.predictions_constant_data_to_xarray()})
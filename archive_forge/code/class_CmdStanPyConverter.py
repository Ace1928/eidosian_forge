import logging
import re
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import numpy as np
from ..rcparams import rcParams
from .base import dict_to_dataset, infer_stan_dtypes, make_attrs, requires
from .inference_data import InferenceData
class CmdStanPyConverter:
    """Encapsulate CmdStanPy specific logic."""

    def __init__(self, *, posterior=None, posterior_predictive=None, predictions=None, prior=None, prior_predictive=None, observed_data=None, constant_data=None, predictions_constant_data=None, log_likelihood=None, index_origin=None, coords=None, dims=None, save_warmup=None, dtypes=None):
        self.posterior = posterior
        self.posterior_predictive = posterior_predictive
        self.predictions = predictions
        self.prior = prior
        self.prior_predictive = prior_predictive
        self.observed_data = observed_data
        self.constant_data = constant_data
        self.predictions_constant_data = predictions_constant_data
        self.log_likelihood = rcParams['data.log_likelihood'] if log_likelihood is None else log_likelihood
        self.index_origin = index_origin
        self.coords = coords
        self.dims = dims
        self.save_warmup = rcParams['data.save_warmup'] if save_warmup is None else save_warmup
        import cmdstanpy
        if dtypes is None:
            dtypes = {}
        elif isinstance(dtypes, cmdstanpy.model.CmdStanModel):
            model_code = dtypes.code()
            dtypes = infer_stan_dtypes(model_code)
        elif isinstance(dtypes, str):
            dtypes_path = Path(dtypes)
            if dtypes_path.exists():
                with dtypes_path.open('r', encoding='UTF-8') as f_obj:
                    model_code = f_obj.read()
            else:
                model_code = dtypes
            dtypes = infer_stan_dtypes(model_code)
        self.dtypes = dtypes
        if hasattr(self.posterior, 'metadata') and hasattr(self.posterior.metadata, 'stan_vars_cols'):
            if self.log_likelihood is True and 'log_lik' in self.posterior.metadata.stan_vars_cols:
                self.log_likelihood = ['log_lik']
        elif hasattr(self.posterior, 'metadata') and hasattr(self.posterior.metadata, 'stan_vars_cols'):
            if self.log_likelihood is True and 'log_lik' in self.posterior.metadata.stan_vars_cols:
                self.log_likelihood = ['log_lik']
        elif hasattr(self.posterior, 'stan_vars_cols'):
            if self.log_likelihood is True and 'log_lik' in self.posterior.stan_vars_cols:
                self.log_likelihood = ['log_lik']
        elif hasattr(self.posterior, 'metadata') and hasattr(self.posterior.metadata, 'stan_vars'):
            if self.log_likelihood is True and 'log_lik' in self.posterior.metadata.stan_vars:
                self.log_likelihood = ['log_lik']
        elif self.log_likelihood is True and self.posterior is not None and hasattr(self.posterior, 'column_names') and any((name.split('[')[0] == 'log_lik' for name in self.posterior.column_names)):
            self.log_likelihood = ['log_lik']
        if isinstance(self.log_likelihood, bool):
            self.log_likelihood = None
        self.cmdstanpy = cmdstanpy

    @requires('posterior')
    def posterior_to_xarray(self):
        """Extract posterior samples from output csv."""
        if not (hasattr(self.posterior, 'metadata') or hasattr(self.posterior, 'stan_vars_cols')):
            return self.posterior_to_xarray_pre_v_0_9_68()
        if hasattr(self.posterior, 'metadata') and hasattr(self.posterior.metadata, 'stan_vars_cols') or hasattr(self.posterior, 'stan_vars_cols'):
            return self.posterior_to_xarray_pre_v_1_0_0()
        if hasattr(self.posterior, 'metadata') and hasattr(self.posterior.metadata, 'stan_vars_cols'):
            return self.posterior_to_xarray_pre_v_1_2_0()
        items = list(self.posterior.metadata.stan_vars)
        if self.posterior_predictive is not None:
            try:
                items = _filter(items, self.posterior_predictive)
            except ValueError:
                pass
        if self.predictions is not None:
            try:
                items = _filter(items, self.predictions)
            except ValueError:
                pass
        if self.log_likelihood is not None:
            try:
                items = _filter(items, self.log_likelihood)
            except ValueError:
                pass
        valid_cols = []
        for item in items:
            if hasattr(self.posterior, 'metadata'):
                if item in self.posterior.metadata.stan_vars:
                    valid_cols.append(item)
        data, data_warmup = _unpack_fit(self.posterior, items, self.save_warmup, self.dtypes)
        dims = deepcopy(self.dims) if self.dims is not None else {}
        coords = deepcopy(self.coords) if self.coords is not None else {}
        return (dict_to_dataset(data, library=self.cmdstanpy, coords=coords, dims=dims, index_origin=self.index_origin), dict_to_dataset(data_warmup, library=self.cmdstanpy, coords=coords, dims=dims, index_origin=self.index_origin))

    @requires('posterior')
    def sample_stats_to_xarray(self):
        """Extract sample_stats from prosterior fit."""
        return self.stats_to_xarray(self.posterior)

    @requires('prior')
    def sample_stats_prior_to_xarray(self):
        """Extract sample_stats from prior fit."""
        return self.stats_to_xarray(self.prior)

    def stats_to_xarray(self, fit):
        """Extract sample_stats from fit."""
        if not (hasattr(fit, 'metadata') or hasattr(fit, 'sampler_vars_cols')):
            return self.sample_stats_to_xarray_pre_v_0_9_68(fit)
        if hasattr(fit, 'metadata') and hasattr(fit.metadata, 'stan_vars_cols') or hasattr(fit, 'stan_vars_cols'):
            return self.sample_stats_to_xarray_pre_v_1_0_0(fit)
        if hasattr(fit, 'metadata') and hasattr(fit.metadata, 'stan_vars_cols'):
            return self.sample_stats_to_xarray_pre_v_1_2_0(fit)
        dtypes = {'divergent__': bool, 'n_leapfrog__': np.int64, 'treedepth__': np.int64, **self.dtypes}
        items = list(fit.method_variables())
        rename_dict = {'divergent': 'diverging', 'n_leapfrog': 'n_steps', 'treedepth': 'tree_depth', 'stepsize': 'step_size', 'accept_stat': 'acceptance_rate'}
        data, data_warmup = _unpack_fit(fit, items, self.save_warmup, self.dtypes)
        for item in items:
            name = re.sub('__$', '', item)
            name = rename_dict.get(name, name)
            data[name] = data.pop(item).astype(dtypes.get(item, float))
            if data_warmup:
                data_warmup[name] = data_warmup.pop(item).astype(dtypes.get(item, float))
        return (dict_to_dataset(data, library=self.cmdstanpy, coords=self.coords, dims=self.dims, index_origin=self.index_origin), dict_to_dataset(data_warmup, library=self.cmdstanpy, coords=self.coords, dims=self.dims, index_origin=self.index_origin))

    @requires('posterior')
    @requires('posterior_predictive')
    def posterior_predictive_to_xarray(self):
        """Convert posterior_predictive samples to xarray."""
        return self.predictive_to_xarray(self.posterior_predictive, self.posterior)

    @requires('prior')
    @requires('prior_predictive')
    def prior_predictive_to_xarray(self):
        """Convert prior_predictive samples to xarray."""
        return self.predictive_to_xarray(self.prior_predictive, self.prior)

    def predictive_to_xarray(self, names, fit):
        """Convert predictive samples to xarray."""
        predictive = _as_set(names)
        if not (hasattr(fit, 'metadata') or hasattr(fit, 'stan_vars_cols')):
            valid_cols = _filter_columns(fit.column_names, predictive)
            data, data_warmup = _unpack_frame(fit, fit.column_names, valid_cols, self.save_warmup, self.dtypes)
        elif hasattr(fit, 'metadata') and hasattr(fit.metadata, 'sample_vars_cols') or hasattr(fit, 'stan_vars_cols'):
            data, data_warmup = _unpack_fit_pre_v_1_0_0(fit, predictive, self.save_warmup, self.dtypes)
        elif hasattr(fit, 'metadata') and hasattr(fit.metadata, 'stan_vars_cols'):
            data, data_warmup = _unpack_fit_pre_v_1_2_0(fit, predictive, self.save_warmup, self.dtypes)
        else:
            data, data_warmup = _unpack_fit(fit, predictive, self.save_warmup, self.dtypes)
        return (dict_to_dataset(data, library=self.cmdstanpy, coords=self.coords, dims=self.dims, index_origin=self.index_origin), dict_to_dataset(data_warmup, library=self.cmdstanpy, coords=self.coords, dims=self.dims, index_origin=self.index_origin))

    @requires('posterior')
    @requires('predictions')
    def predictions_to_xarray(self):
        """Convert out of sample predictions samples to xarray."""
        predictions = _as_set(self.predictions)
        if not (hasattr(self.posterior, 'metadata') or hasattr(self.posterior, 'stan_vars_cols')):
            columns = self.posterior.column_names
            valid_cols = _filter_columns(columns, predictions)
            data, data_warmup = _unpack_frame(self.posterior, columns, valid_cols, self.save_warmup, self.dtypes)
        elif hasattr(self.posterior, 'metadata') and hasattr(self.posterior.metadata, 'sample_vars_cols') or hasattr(self.posterior, 'stan_vars_cols'):
            data, data_warmup = _unpack_fit_pre_v_1_0_0(self.posterior, predictions, self.save_warmup, self.dtypes)
        elif hasattr(self.posterior, 'metadata') and hasattr(self.posterior.metadata, 'stan_vars_cols'):
            data, data_warmup = _unpack_fit_pre_v_1_2_0(self.posterior, predictions, self.save_warmup, self.dtypes)
        else:
            data, data_warmup = _unpack_fit(self.posterior, predictions, self.save_warmup, self.dtypes)
        return (dict_to_dataset(data, library=self.cmdstanpy, coords=self.coords, dims=self.dims, index_origin=self.index_origin), dict_to_dataset(data_warmup, library=self.cmdstanpy, coords=self.coords, dims=self.dims, index_origin=self.index_origin))

    @requires('posterior')
    @requires('log_likelihood')
    def log_likelihood_to_xarray(self):
        """Convert elementwise log likelihood samples to xarray."""
        log_likelihood = _as_set(self.log_likelihood)
        if not (hasattr(self.posterior, 'metadata') or hasattr(self.posterior, 'stan_vars_cols')):
            columns = self.posterior.column_names
            valid_cols = _filter_columns(columns, log_likelihood)
            data, data_warmup = _unpack_frame(self.posterior, columns, valid_cols, self.save_warmup, self.dtypes)
        elif hasattr(self.posterior, 'metadata') and hasattr(self.posterior.metadata, 'sample_vars_cols') or hasattr(self.posterior, 'stan_vars_cols'):
            data, data_warmup = _unpack_fit_pre_v_1_0_0(self.posterior, log_likelihood, self.save_warmup, self.dtypes)
        elif hasattr(self.posterior, 'metadata') and hasattr(self.posterior.metadata, 'stan_vars_cols'):
            data, data_warmup = _unpack_fit_pre_v_1_2_0(self.posterior, log_likelihood, self.save_warmup, self.dtypes)
        else:
            data, data_warmup = _unpack_fit(self.posterior, log_likelihood, self.save_warmup, self.dtypes)
        if isinstance(self.log_likelihood, dict):
            data = {obs_name: data[lik_name] for obs_name, lik_name in self.log_likelihood.items()}
            if data_warmup:
                data_warmup = {obs_name: data_warmup[lik_name] for obs_name, lik_name in self.log_likelihood.items()}
        return (dict_to_dataset(data, library=self.cmdstanpy, coords=self.coords, dims=self.dims, index_origin=self.index_origin, skip_event_dims=True), dict_to_dataset(data_warmup, library=self.cmdstanpy, coords=self.coords, dims=self.dims, index_origin=self.index_origin, skip_event_dims=True))

    @requires('prior')
    def prior_to_xarray(self):
        """Convert prior samples to xarray."""
        if not (hasattr(self.prior, 'metadata') or hasattr(self.prior, 'stan_vars_cols')):
            columns = self.prior.column_names
            prior_predictive = _as_set(self.prior_predictive)
            prior_predictive = _filter_columns(columns, prior_predictive)
            invalid_cols = set(prior_predictive + [col for col in columns if col.endswith('__')])
            valid_cols = [col for col in columns if col not in invalid_cols]
            data, data_warmup = _unpack_frame(self.prior, columns, valid_cols, self.save_warmup, self.dtypes)
        elif hasattr(self.prior, 'metadata') and hasattr(self.prior.metadata, 'sample_vars_cols') or hasattr(self.prior, 'stan_vars_cols'):
            if hasattr(self.prior, 'metadata'):
                items = list(self.prior.metadata.stan_vars_cols.keys())
            else:
                items = list(self.prior.stan_vars_cols.keys())
            if self.prior_predictive is not None:
                try:
                    items = _filter(items, self.prior_predictive)
                except ValueError:
                    pass
            data, data_warmup = _unpack_fit_pre_v_1_0_0(self.prior, items, self.save_warmup, self.dtypes)
        elif hasattr(self.prior, 'metadata') and hasattr(self.prior.metadata, 'stan_vars_cols'):
            items = list(self.prior.metadata.stan_vars_cols.keys())
            if self.prior_predictive is not None:
                try:
                    items = _filter(items, self.prior_predictive)
                except ValueError:
                    pass
            data, data_warmup = _unpack_fit_pre_v_1_2_0(self.prior, items, self.save_warmup, self.dtypes)
        else:
            items = list(self.prior.metadata.stan_vars.keys())
            if self.prior_predictive is not None:
                try:
                    items = _filter(items, self.prior_predictive)
                except ValueError:
                    pass
            data, data_warmup = _unpack_fit(self.prior, items, self.save_warmup, self.dtypes)
        return (dict_to_dataset(data, library=self.cmdstanpy, coords=self.coords, dims=self.dims, index_origin=self.index_origin), dict_to_dataset(data_warmup, library=self.cmdstanpy, coords=self.coords, dims=self.dims, index_origin=self.index_origin))

    @requires('observed_data')
    def observed_data_to_xarray(self):
        """Convert observed data to xarray."""
        return dict_to_dataset(self.observed_data, library=self.cmdstanpy, coords=self.coords, dims=self.dims, default_dims=[], index_origin=self.index_origin)

    @requires('constant_data')
    def constant_data_to_xarray(self):
        """Convert constant data to xarray."""
        return dict_to_dataset(self.constant_data, library=self.cmdstanpy, coords=self.coords, dims=self.dims, default_dims=[], index_origin=self.index_origin)

    @requires('predictions_constant_data')
    def predictions_constant_data_to_xarray(self):
        """Convert constant data to xarray."""
        return dict_to_dataset(self.predictions_constant_data, library=self.cmdstanpy, coords=self.coords, dims=self.dims, attrs=make_attrs(library=self.cmdstanpy), default_dims=[], index_origin=self.index_origin)

    def to_inference_data(self):
        """Convert all available data to an InferenceData object.

        Note that if groups can not be created (i.e., there is no `output`, so
        the `posterior` and `sample_stats` can not be extracted), then the InferenceData
        will not have those groups.
        """
        return InferenceData(save_warmup=self.save_warmup, **{'posterior': self.posterior_to_xarray(), 'sample_stats': self.sample_stats_to_xarray(), 'posterior_predictive': self.posterior_predictive_to_xarray(), 'predictions': self.predictions_to_xarray(), 'prior': self.prior_to_xarray(), 'sample_stats_prior': self.sample_stats_prior_to_xarray(), 'prior_predictive': self.prior_predictive_to_xarray(), 'observed_data': self.observed_data_to_xarray(), 'constant_data': self.constant_data_to_xarray(), 'predictions_constant_data': self.predictions_constant_data_to_xarray(), 'log_likelihood': self.log_likelihood_to_xarray()})

    def posterior_to_xarray_pre_v_1_2_0(self):
        items = list(self.posterior.metadata.stan_vars_cols)
        if self.posterior_predictive is not None:
            try:
                items = _filter(items, self.posterior_predictive)
            except ValueError:
                pass
        if self.predictions is not None:
            try:
                items = _filter(items, self.predictions)
            except ValueError:
                pass
        if self.log_likelihood is not None:
            try:
                items = _filter(items, self.log_likelihood)
            except ValueError:
                pass
        valid_cols = []
        for item in items:
            if hasattr(self.posterior, 'metadata'):
                if item in self.posterior.metadata.stan_vars_cols:
                    valid_cols.append(item)
        data, data_warmup = _unpack_fit_pre_v_1_2_0(self.posterior, items, self.save_warmup, self.dtypes)
        dims = deepcopy(self.dims) if self.dims is not None else {}
        coords = deepcopy(self.coords) if self.coords is not None else {}
        return (dict_to_dataset(data, library=self.cmdstanpy, coords=coords, dims=dims, index_origin=self.index_origin), dict_to_dataset(data_warmup, library=self.cmdstanpy, coords=coords, dims=dims, index_origin=self.index_origin))

    @requires('posterior')
    def posterior_to_xarray_pre_v_1_0_0(self):
        if hasattr(self.posterior, 'metadata'):
            items = list(self.posterior.metadata.stan_vars_cols.keys())
        else:
            items = list(self.posterior.stan_vars_cols.keys())
        if self.posterior_predictive is not None:
            try:
                items = _filter(items, self.posterior_predictive)
            except ValueError:
                pass
        if self.predictions is not None:
            try:
                items = _filter(items, self.predictions)
            except ValueError:
                pass
        if self.log_likelihood is not None:
            try:
                items = _filter(items, self.log_likelihood)
            except ValueError:
                pass
        valid_cols = []
        for item in items:
            if hasattr(self.posterior, 'metadata'):
                valid_cols.extend(self.posterior.metadata.stan_vars_cols[item])
            else:
                valid_cols.extend(self.posterior.stan_vars_cols[item])
        data, data_warmup = _unpack_fit_pre_v_1_0_0(self.posterior, items, self.save_warmup, self.dtypes)
        dims = deepcopy(self.dims) if self.dims is not None else {}
        coords = deepcopy(self.coords) if self.coords is not None else {}
        return (dict_to_dataset(data, library=self.cmdstanpy, coords=coords, dims=dims, index_origin=self.index_origin), dict_to_dataset(data_warmup, library=self.cmdstanpy, coords=coords, dims=dims, index_origin=self.index_origin))

    @requires('posterior')
    def posterior_to_xarray_pre_v_0_9_68(self):
        """Extract posterior samples from output csv."""
        columns = self.posterior.column_names
        posterior_predictive = self.posterior_predictive
        if posterior_predictive is None:
            posterior_predictive = []
        elif isinstance(posterior_predictive, str):
            posterior_predictive = [col for col in columns if posterior_predictive == col.split('[')[0].split('.')[0]]
        else:
            posterior_predictive = [col for col in columns if any((item == col.split('[')[0].split('.')[0] for item in posterior_predictive))]
        predictions = self.predictions
        if predictions is None:
            predictions = []
        elif isinstance(predictions, str):
            predictions = [col for col in columns if predictions == col.split('[')[0].split('.')[0]]
        else:
            predictions = [col for col in columns if any((item == col.split('[')[0].split('.')[0] for item in predictions))]
        log_likelihood = self.log_likelihood
        if log_likelihood is None:
            log_likelihood = []
        elif isinstance(log_likelihood, str):
            log_likelihood = [col for col in columns if log_likelihood == col.split('[')[0].split('.')[0]]
        else:
            log_likelihood = [col for col in columns if any((item == col.split('[')[0].split('.')[0] for item in log_likelihood))]
        invalid_cols = set(posterior_predictive + predictions + log_likelihood + [col for col in columns if col.endswith('__')])
        valid_cols = [col for col in columns if col not in invalid_cols]
        data, data_warmup = _unpack_frame(self.posterior, columns, valid_cols, self.save_warmup, self.dtypes)
        return (dict_to_dataset(data, library=self.cmdstanpy, coords=self.coords, dims=self.dims, index_origin=self.index_origin), dict_to_dataset(data_warmup, library=self.cmdstanpy, coords=self.coords, dims=self.dims, index_origin=self.index_origin))

    def sample_stats_to_xarray_pre_v_1_2_0(self, fit):
        dtypes = {'divergent__': bool, 'n_leapfrog__': np.int64, 'treedepth__': np.int64, **self.dtypes}
        items = list(fit.metadata.method_vars_cols.keys())
        rename_dict = {'divergent': 'diverging', 'n_leapfrog': 'n_steps', 'treedepth': 'tree_depth', 'stepsize': 'step_size', 'accept_stat': 'acceptance_rate'}
        data, data_warmup = _unpack_fit_pre_v_1_2_0(fit, items, self.save_warmup, self.dtypes)
        for item in items:
            name = re.sub('__$', '', item)
            name = rename_dict.get(name, name)
            data[name] = data.pop(item).astype(dtypes.get(item, float))
            if data_warmup:
                data_warmup[name] = data_warmup.pop(item).astype(dtypes.get(item, float))
        return (dict_to_dataset(data, library=self.cmdstanpy, coords=self.coords, dims=self.dims, index_origin=self.index_origin), dict_to_dataset(data_warmup, library=self.cmdstanpy, coords=self.coords, dims=self.dims, index_origin=self.index_origin))

    def sample_stats_to_xarray_pre_v_1_0_0(self, fit):
        """Extract sample_stats from fit."""
        dtypes = {'divergent__': bool, 'n_leapfrog__': np.int64, 'treedepth__': np.int64, **self.dtypes}
        if hasattr(fit, 'metadata'):
            items = list(fit.metadata._method_vars_cols.keys())
        else:
            items = list(fit.sampler_vars_cols.keys())
        rename_dict = {'divergent': 'diverging', 'n_leapfrog': 'n_steps', 'treedepth': 'tree_depth', 'stepsize': 'step_size', 'accept_stat': 'acceptance_rate'}
        data, data_warmup = _unpack_fit_pre_v_1_0_0(fit, items, self.save_warmup, self.dtypes)
        for item in items:
            name = re.sub('__$', '', item)
            name = rename_dict.get(name, name)
            data[name] = data.pop(item).astype(dtypes.get(item, float))
            if data_warmup:
                data_warmup[name] = data_warmup.pop(item).astype(dtypes.get(item, float))
        return (dict_to_dataset(data, library=self.cmdstanpy, coords=self.coords, dims=self.dims, index_origin=self.index_origin), dict_to_dataset(data_warmup, library=self.cmdstanpy, coords=self.coords, dims=self.dims, index_origin=self.index_origin))

    def sample_stats_to_xarray_pre_v_0_9_68(self, fit):
        """Extract sample_stats from fit."""
        dtypes = {'divergent__': bool, 'n_leapfrog__': np.int64, 'treedepth__': np.int64}
        columns = fit.column_names
        valid_cols = [col for col in columns if col.endswith('__')]
        data, data_warmup = _unpack_frame(fit, columns, valid_cols, self.save_warmup, self.dtypes)
        for s_param in list(data.keys()):
            s_param_, *_ = s_param.split('.')
            name = re.sub('__$', '', s_param_)
            name = 'diverging' if name == 'divergent' else name
            data[name] = data.pop(s_param).astype(dtypes.get(s_param, float))
            if data_warmup:
                data_warmup[name] = data_warmup.pop(s_param).astype(dtypes.get(s_param, float))
        return (dict_to_dataset(data, library=self.cmdstanpy, coords=self.coords, dims=self.dims, index_origin=self.index_origin), dict_to_dataset(data_warmup, library=self.cmdstanpy, coords=self.coords, dims=self.dims, index_origin=self.index_origin))
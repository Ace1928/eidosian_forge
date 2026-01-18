import re
from collections import OrderedDict
from copy import deepcopy
from math import ceil
import numpy as np
import xarray as xr
from .. import _log
from ..rcparams import rcParams
from .base import dict_to_dataset, generate_dims_coords, infer_stan_dtypes, make_attrs, requires
from .inference_data import InferenceData
class PyStanConverter:
    """Encapsulate PyStan specific logic."""

    def __init__(self, *, posterior=None, posterior_predictive=None, predictions=None, prior=None, prior_predictive=None, observed_data=None, constant_data=None, predictions_constant_data=None, log_likelihood=None, coords=None, dims=None, save_warmup=None, dtypes=None):
        self.posterior = posterior
        self.posterior_predictive = posterior_predictive
        self.predictions = predictions
        self.prior = prior
        self.prior_predictive = prior_predictive
        self.observed_data = observed_data
        self.constant_data = constant_data
        self.predictions_constant_data = predictions_constant_data
        self.log_likelihood = rcParams['data.log_likelihood'] if log_likelihood is None else log_likelihood
        self.coords = coords
        self.dims = dims
        self.save_warmup = rcParams['data.save_warmup'] if save_warmup is None else save_warmup
        self.dtypes = dtypes
        if self.log_likelihood is True and self.posterior is not None and ('log_lik' in self.posterior.sim['pars_oi']):
            self.log_likelihood = ['log_lik']
        elif isinstance(self.log_likelihood, bool):
            self.log_likelihood = None
        import pystan
        self.pystan = pystan

    @requires('posterior')
    def posterior_to_xarray(self):
        """Extract posterior samples from fit."""
        posterior = self.posterior
        posterior_predictive = self.posterior_predictive
        if posterior_predictive is None:
            posterior_predictive = []
        elif isinstance(posterior_predictive, str):
            posterior_predictive = [posterior_predictive]
        predictions = self.predictions
        if predictions is None:
            predictions = []
        elif isinstance(predictions, str):
            predictions = [predictions]
        log_likelihood = self.log_likelihood
        if log_likelihood is None:
            log_likelihood = []
        elif isinstance(log_likelihood, str):
            log_likelihood = [log_likelihood]
        elif isinstance(log_likelihood, dict):
            log_likelihood = list(log_likelihood.values())
        ignore = posterior_predictive + predictions + log_likelihood + ['lp__']
        data, data_warmup = get_draws(posterior, ignore=ignore, warmup=self.save_warmup, dtypes=self.dtypes)
        attrs = get_attrs(posterior)
        return (dict_to_dataset(data, library=self.pystan, attrs=attrs, coords=self.coords, dims=self.dims), dict_to_dataset(data_warmup, library=self.pystan, attrs=attrs, coords=self.coords, dims=self.dims))

    @requires('posterior')
    def sample_stats_to_xarray(self):
        """Extract sample_stats from posterior."""
        posterior = self.posterior
        data, data_warmup = get_sample_stats(posterior, warmup=self.save_warmup)
        stat_lp, stat_lp_warmup = get_draws(posterior, variables='lp__', warmup=self.save_warmup, dtypes=self.dtypes)
        data['lp'] = stat_lp['lp__']
        if stat_lp_warmup:
            data_warmup['lp'] = stat_lp_warmup['lp__']
        attrs = get_attrs(posterior)
        return (dict_to_dataset(data, library=self.pystan, attrs=attrs, coords=self.coords, dims=self.dims), dict_to_dataset(data_warmup, library=self.pystan, attrs=attrs, coords=self.coords, dims=self.dims))

    @requires('posterior')
    @requires('log_likelihood')
    def log_likelihood_to_xarray(self):
        """Store log_likelihood data in log_likelihood group."""
        fit = self.posterior
        log_likelihood = self.log_likelihood
        if isinstance(log_likelihood, str):
            log_likelihood = [log_likelihood]
        if isinstance(log_likelihood, (list, tuple)):
            log_likelihood = {name: name for name in log_likelihood}
        log_likelihood_draws, log_likelihood_draws_warmup = get_draws(fit, variables=list(log_likelihood.values()), warmup=self.save_warmup, dtypes=self.dtypes)
        data = {obs_var_name: log_likelihood_draws[log_like_name] for obs_var_name, log_like_name in log_likelihood.items() if log_like_name in log_likelihood_draws}
        data_warmup = {obs_var_name: log_likelihood_draws_warmup[log_like_name] for obs_var_name, log_like_name in log_likelihood.items() if log_like_name in log_likelihood_draws_warmup}
        return (dict_to_dataset(data, library=self.pystan, coords=self.coords, dims=self.dims, skip_event_dims=True), dict_to_dataset(data_warmup, library=self.pystan, coords=self.coords, dims=self.dims, skip_event_dims=True))

    @requires('posterior')
    @requires('posterior_predictive')
    def posterior_predictive_to_xarray(self):
        """Convert posterior_predictive samples to xarray."""
        posterior = self.posterior
        posterior_predictive = self.posterior_predictive
        data, data_warmup = get_draws(posterior, variables=posterior_predictive, warmup=self.save_warmup, dtypes=self.dtypes)
        return (dict_to_dataset(data, library=self.pystan, coords=self.coords, dims=self.dims), dict_to_dataset(data_warmup, library=self.pystan, coords=self.coords, dims=self.dims))

    @requires('posterior')
    @requires('predictions')
    def predictions_to_xarray(self):
        """Convert predictions samples to xarray."""
        posterior = self.posterior
        predictions = self.predictions
        data, data_warmup = get_draws(posterior, variables=predictions, warmup=self.save_warmup, dtypes=self.dtypes)
        return (dict_to_dataset(data, library=self.pystan, coords=self.coords, dims=self.dims), dict_to_dataset(data_warmup, library=self.pystan, coords=self.coords, dims=self.dims))

    @requires('prior')
    def prior_to_xarray(self):
        """Convert prior samples to xarray."""
        prior = self.prior
        prior_predictive = self.prior_predictive
        if prior_predictive is None:
            prior_predictive = []
        elif isinstance(prior_predictive, str):
            prior_predictive = [prior_predictive]
        ignore = prior_predictive + ['lp__']
        data, _ = get_draws(prior, ignore=ignore, warmup=False, dtypes=self.dtypes)
        attrs = get_attrs(prior)
        return dict_to_dataset(data, library=self.pystan, attrs=attrs, coords=self.coords, dims=self.dims)

    @requires('prior')
    def sample_stats_prior_to_xarray(self):
        """Extract sample_stats_prior from prior."""
        prior = self.prior
        data, _ = get_sample_stats(prior, warmup=False)
        stat_lp, _ = get_draws(prior, variables='lp__', warmup=False, dtypes=self.dtypes)
        data['lp'] = stat_lp['lp__']
        attrs = get_attrs(prior)
        return dict_to_dataset(data, library=self.pystan, attrs=attrs, coords=self.coords, dims=self.dims)

    @requires('prior')
    @requires('prior_predictive')
    def prior_predictive_to_xarray(self):
        """Convert prior_predictive samples to xarray."""
        prior = self.prior
        prior_predictive = self.prior_predictive
        data, _ = get_draws(prior, variables=prior_predictive, warmup=False, dtypes=self.dtypes)
        return dict_to_dataset(data, library=self.pystan, coords=self.coords, dims=self.dims)

    @requires('posterior')
    @requires(['observed_data', 'constant_data', 'predictions_constant_data'])
    def data_to_xarray(self):
        """Convert observed, constant data and predictions constant data to xarray."""
        posterior = self.posterior
        dims = {} if self.dims is None else self.dims
        obs_const_dict = {}
        for group_name in ('observed_data', 'constant_data', 'predictions_constant_data'):
            names = getattr(self, group_name)
            if names is None:
                continue
            names = [names] if isinstance(names, str) else names
            data = OrderedDict()
            for key in names:
                vals = np.atleast_1d(posterior.data[key])
                val_dims = dims.get(key)
                val_dims, coords = generate_dims_coords(vals.shape, key, dims=val_dims, coords=self.coords)
                data[key] = xr.DataArray(vals, dims=val_dims, coords=coords)
            obs_const_dict[group_name] = xr.Dataset(data_vars=data, attrs=make_attrs(library=self.pystan))
        return obs_const_dict

    def to_inference_data(self):
        """Convert all available data to an InferenceData object.

        Note that if groups can not be created (i.e., there is no `fit`, so
        the `posterior` and `sample_stats` can not be extracted), then the InferenceData
        will not have those groups.
        """
        data_dict = self.data_to_xarray()
        return InferenceData(save_warmup=self.save_warmup, **{'posterior': self.posterior_to_xarray(), 'sample_stats': self.sample_stats_to_xarray(), 'log_likelihood': self.log_likelihood_to_xarray(), 'posterior_predictive': self.posterior_predictive_to_xarray(), 'predictions': self.predictions_to_xarray(), 'prior': self.prior_to_xarray(), 'sample_stats_prior': self.sample_stats_prior_to_xarray(), 'prior_predictive': self.prior_predictive_to_xarray(), **({} if data_dict is None else data_dict)})
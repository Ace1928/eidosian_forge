import copy
import numpy as np
from typing import Dict, List, Optional, Union
from ray import cloudpickle
from ray.tune.result import DEFAULT_METRIC
from ray.tune.search.sample import (
from ray.tune.search import (
from ray.tune.search.variant_generator import parse_spec_vars
from ray.tune.utils.util import flatten_dict, unflatten_list_dict
import logging
def _setup_experiment(self):
    if self._metric is None and self._mode:
        self._metric = DEFAULT_METRIC
    if not self._ax:
        self._ax = AxClient(**self._ax_kwargs)
    try:
        exp = self._ax.experiment
        has_experiment = True
    except ValueError:
        has_experiment = False
    if not has_experiment:
        if not self._space:
            raise ValueError('You have to create an Ax experiment by calling `AxClient.create_experiment()`, or you should pass an Ax search space as the `space` parameter to `AxSearch`, or pass a `param_space` dict to `tune.Tuner()`.')
        if self._mode not in ['min', 'max']:
            raise ValueError('Please specify the `mode` argument when initializing the `AxSearch` object or pass it to `tune.TuneConfig()`.')
        self._ax.create_experiment(parameters=self._space, objective_name=self._metric, parameter_constraints=self._parameter_constraints, outcome_constraints=self._outcome_constraints, minimize=self._mode != 'max')
    elif any([self._space, self._parameter_constraints, self._outcome_constraints, self._mode, self._metric]):
        raise ValueError('If you create the Ax experiment yourself, do not pass values for these parameters to `AxSearch`: {}.'.format(['space', 'parameter_constraints', 'outcome_constraints', 'mode', 'metric']))
    exp = self._ax.experiment
    self._mode = 'min' if exp.optimization_config.objective.minimize else 'max'
    self._metric = exp.optimization_config.objective.metric.name
    self._parameters = list(exp.parameters)
    if self._ax._enforce_sequential_optimization:
        logger.warning('Detected sequential enforcement. Be sure to use a ConcurrencyLimiter.')
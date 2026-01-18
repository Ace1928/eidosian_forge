from typing import Any, Dict, List, Optional
import numpy as np
import copy
import logging
from functools import partial
from ray import cloudpickle
from ray.tune.result import DEFAULT_METRIC
from ray.tune.search.sample import (
from ray.tune.search import (
from ray.tune.search.variant_generator import assign_value, parse_spec_vars
from ray.tune.utils import flatten_dict
from ray.tune.error import TuneError
def _setup_hyperopt(self) -> None:
    from hyperopt.fmin import generate_trials_to_calculate
    if not self._space:
        raise RuntimeError(UNDEFINED_SEARCH_SPACE.format(cls=self.__class__.__name__, space='space') + HYPEROPT_UNDEFINED_DETAILS)
    if self._metric is None and self._mode:
        self._metric = DEFAULT_METRIC
    if self._points_to_evaluate is None:
        self._hpopt_trials = hpo.Trials()
        self._points_to_evaluate = 0
    else:
        assert isinstance(self._points_to_evaluate, (list, tuple))
        for i in range(len(self._points_to_evaluate)):
            config = self._points_to_evaluate[i]
            self._convert_categories_to_indices(config)
        self._points_to_evaluate = list(reversed(self._points_to_evaluate))
        self._hpopt_trials = generate_trials_to_calculate(self._points_to_evaluate)
        self._hpopt_trials.refresh()
        self._points_to_evaluate = len(self._points_to_evaluate)
    self.domain = hpo.Domain(lambda spc: spc, self._space)
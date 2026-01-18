import copy
import logging
import numpy as np
import pickle
from typing import Dict, List, Optional, Tuple, Union, Any
from ray.tune.result import DEFAULT_METRIC
from ray.tune.search.sample import (
from ray.tune.search import (
from ray.tune.search.variant_generator import parse_spec_vars
from ray.tune.utils import flatten_dict
from ray.tune.utils.util import is_nan_or_inf, unflatten_dict, validate_warmstart
def _setup_skopt(self):
    if self._points_to_evaluate and isinstance(self._points_to_evaluate, list):
        if isinstance(self._points_to_evaluate[0], list):
            self._points_to_evaluate = [dict(zip(self._parameter_names, point)) for point in self._points_to_evaluate]
    validate_warmstart(self._parameter_names, self._points_to_evaluate, self._evaluated_rewards)
    if not self._skopt_opt:
        if not self._space:
            raise ValueError("If you don't pass an optimizer instance to SkOptSearch, pass a valid `space` parameter.")
        self._skopt_opt = sko.Optimizer(self._parameter_ranges)
    if self._points_to_evaluate and self._evaluated_rewards:
        skopt_points = [[point[par] for par in self._parameter_names] for point in self._points_to_evaluate]
        self._skopt_opt.tell(skopt_points, self._evaluated_rewards)
    elif self._points_to_evaluate:
        self._initial_points = self._points_to_evaluate
    self._parameters = self._parameter_names
    if self._mode == 'max':
        self._metric_op = -1.0
    elif self._mode == 'min':
        self._metric_op = 1.0
    if self._metric is None and self._mode:
        self._metric = DEFAULT_METRIC
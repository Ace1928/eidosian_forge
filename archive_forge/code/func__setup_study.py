import time
import logging
import pickle
import functools
import warnings
from packaging import version
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from ray.air.constants import TRAINING_ITERATION
from ray.tune.result import DEFAULT_METRIC
from ray.tune.search.sample import (
from ray.tune.search import (
from ray.tune.search.variant_generator import parse_spec_vars
from ray.tune.utils.util import flatten_dict, unflatten_dict, validate_warmstart
def _setup_study(self, mode: Union[str, list]):
    if self._metric is None and self._mode:
        if isinstance(self._mode, list):
            raise ValueError('If ``mode`` is a list (multi-objective optimization case), ``metric`` must be defined.')
        self._metric = DEFAULT_METRIC
    pruner = ot.pruners.NopPruner()
    storage = ot.storages.InMemoryStorage()
    if self._sampler:
        sampler = self._sampler
    elif isinstance(mode, list) and version.parse(ot.__version__) < version.parse('2.9.0'):
        sampler = ot.samplers.MOTPESampler(seed=self._seed)
    else:
        sampler = ot.samplers.TPESampler(seed=self._seed)
    if isinstance(mode, list):
        study_direction_args = dict(directions=['minimize' if m == 'min' else 'maximize' for m in mode])
    else:
        study_direction_args = dict(direction='minimize' if mode == 'min' else 'maximize')
    self._ot_study = ot.study.create_study(storage=storage, sampler=sampler, pruner=pruner, study_name=self._study_name, load_if_exists=True, **study_direction_args)
    if self._points_to_evaluate:
        validate_warmstart(self._space, self._points_to_evaluate, self._evaluated_rewards, validate_point_name_lengths=not callable(self._space))
        if self._evaluated_rewards:
            for point, reward in zip(self._points_to_evaluate, self._evaluated_rewards):
                self.add_evaluated_point(point, reward)
        else:
            for point in self._points_to_evaluate:
                self._ot_study.enqueue_trial(point)
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
def add_evaluated_point(self, parameters: Dict, value: float, error: bool=False, pruned: bool=False, intermediate_values: Optional[List[float]]=None):
    if not self._space:
        raise RuntimeError(UNDEFINED_SEARCH_SPACE.format(cls=self.__class__.__name__, space='space'))
    if not self._metric or not self._mode:
        raise RuntimeError(UNDEFINED_METRIC_MODE.format(cls=self.__class__.__name__, metric=self._metric, mode=self._mode))
    if callable(self._space):
        raise TypeError('Define-by-run function passed in `space` argument is not yet supported when using `evaluated_rewards`. Please provide an `OptunaDistribution` dict or pass a Ray Tune search space to `tune.Tuner()`.')
    ot_trial_state = OptunaTrialState.COMPLETE
    if error:
        ot_trial_state = OptunaTrialState.FAIL
    elif pruned:
        ot_trial_state = OptunaTrialState.PRUNED
    if intermediate_values:
        intermediate_values_dict = {i: value for i, value in enumerate(intermediate_values)}
    else:
        intermediate_values_dict = None
    trial = ot.trial.create_trial(state=ot_trial_state, value=value, params=parameters, distributions=self._space, intermediate_values=intermediate_values_dict)
    self._ot_study.add_trial(trial)
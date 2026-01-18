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
def _process_result(self, trial_id: str, result: Dict):
    skopt_trial_info = self._live_trial_mapping[trial_id]
    if result and (not is_nan_or_inf(result[self._metric])):
        self._skopt_opt.tell(skopt_trial_info, self._metric_op * result[self._metric])
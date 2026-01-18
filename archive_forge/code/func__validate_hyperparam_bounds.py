from typing import Callable, Dict, Optional, Tuple, Union, TYPE_CHECKING
from copy import deepcopy
import logging
import numpy as np
import pandas as pd
from ray.tune import TuneError
from ray.tune.experiment import Trial
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.schedulers.pbt import _PBTTrialState
from ray.tune.utils.util import flatten_dict, unflatten_dict
from ray.util.debug import log_once
def _validate_hyperparam_bounds(self, hyperparam_bounds: dict):
    """Check that each hyperparam bound is of the form [low, high].

        Raises:
            ValueError: if any of the hyperparam bounds are of an invalid format.
        """
    for key, value in hyperparam_bounds.items():
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(f"`hyperparam_bounds` values must either be a list or tuple of size 2, but got {value} instead for the param '{key}'")
        low, high = value
        if low > high:
            raise ValueError(f"`hyperparam_bounds` values must be of the form [low, high] where low <= high, but got {value} instead for param '{key}'.")
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
class _OptunaTrialSuggestCaptor:
    """Utility to capture returned values from Optuna's suggest_ methods.

    This will wrap around the ``optuna.Trial` object and decorate all
    `suggest_` callables with a function capturing the returned value,
    which will be saved in the ``captured_values`` dict.
    """

    def __init__(self, ot_trial: OptunaTrial) -> None:
        self.ot_trial = ot_trial
        self.captured_values: Dict[str, Any] = {}

    def _get_wrapper(self, func: Callable) -> Callable:

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            name = kwargs.get('name', args[0])
            ret = func(*args, **kwargs)
            self.captured_values[name] = ret
            return ret
        return wrapper

    def __getattr__(self, item_name: str) -> Any:
        item = getattr(self.ot_trial, item_name)
        if item_name.startswith('suggest_') and callable(item):
            return self._get_wrapper(item)
        return item
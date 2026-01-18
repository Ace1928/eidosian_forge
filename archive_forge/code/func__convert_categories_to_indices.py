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
def _convert_categories_to_indices(self, config) -> None:
    """Convert config parameters for categories into hyperopt-compatible
        representations where instead the index of the category is expected."""

    def _lookup(config_dict, space_dict, key):
        if isinstance(config_dict[key], dict):
            for k in config_dict[key]:
                _lookup(config_dict[key], space_dict[key], k)
        elif key in space_dict and isinstance(space_dict[key], hpo.base.pyll.Apply) and (space_dict[key].name == 'switch'):
            if len(space_dict[key].pos_args) > 0:
                categories = [a.obj for a in space_dict[key].pos_args[1:] if a.name == 'literal']
                try:
                    idx = categories.index(config_dict[key])
                except ValueError as exc:
                    msg = f'Did not find category with value `{config_dict[key]}` in hyperopt parameter `{key}`. '
                    if isinstance(config_dict[key], int):
                        msg += 'In previous versions, a numerical index was expected for categorical values of `points_to_evaluate`, but in ray>=1.2.0, the categorical value is expected to be directly provided. '
                    msg += 'Please make sure the specified category is valid.'
                    raise ValueError(msg) from exc
                config_dict[key] = idx
    for k in config:
        _lookup(config, self._space, k)
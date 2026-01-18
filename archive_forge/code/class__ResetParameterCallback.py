from collections import OrderedDict
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union
from .basic import (Booster, _ConfigAliases, _LGBM_BoosterEvalMethodResultType,
class _ResetParameterCallback:
    """Internal reset parameter callable class."""

    def __init__(self, **kwargs: Union[list, Callable]) -> None:
        self.order = 10
        self.before_iteration = True
        self.kwargs = kwargs

    def __call__(self, env: CallbackEnv) -> None:
        new_parameters = {}
        for key, value in self.kwargs.items():
            if isinstance(value, list):
                if len(value) != env.end_iteration - env.begin_iteration:
                    raise ValueError(f"Length of list {key!r} has to be equal to 'num_boost_round'.")
                new_param = value[env.iteration - env.begin_iteration]
            elif callable(value):
                new_param = value(env.iteration - env.begin_iteration)
            else:
                raise ValueError('Only list and callable values are supported as a mapping from boosting round index to new parameter value.')
            if new_param != env.params.get(key, None):
                new_parameters[key] = new_param
        if new_parameters:
            if isinstance(env.model, Booster):
                env.model.reset_parameter(new_parameters)
            else:
                for booster in env.model.boosters:
                    booster.reset_parameter(new_parameters)
            env.params.update(new_parameters)
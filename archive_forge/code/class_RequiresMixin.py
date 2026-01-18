import os
from functools import wraps
from typing import Any, Callable, Dict, TypeVar, cast
class RequiresMixin:
    requirement = ''

    def __init__(self) -> None:
        self._check_if_requirements_met()

    def __post_init__(self) -> None:
        self._check_if_requirements_met()

    def _check_if_requirements_met(self) -> None:
        env_var = requirement_env_var_mapping[self.requirement]
        if not os.getenv(env_var):
            raise Exception(f'You must explicitly enable this feature with `wandb.require("{self.requirement})"')
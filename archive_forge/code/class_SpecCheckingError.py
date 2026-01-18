import functools
import logging
from collections import abc
from typing import Union, Mapping, Any, Callable
from ray.rllib.core.models.specs.specs_base import Spec, TypeSpec
from ray.rllib.core.models.specs.specs_dict import SpecDict
from ray.rllib.core.models.specs.typing import SpecType
from ray.rllib.utils.nested_dict import NestedDict
from ray.util.annotations import DeveloperAPI
@DeveloperAPI
class SpecCheckingError(Exception):
    """Raised when there is an error in the spec checking.

    This Error is raised when inputs or outputs do match the defined specs.
    """
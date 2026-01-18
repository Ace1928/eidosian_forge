import inspect
import logging
import types
from typing import Any, Callable, Dict, Optional, Type, Union, TYPE_CHECKING
import ray
from ray.tune.execution.placement_groups import (
from ray.air.config import ScalingConfig
from ray.tune.registry import _ParameterRegistry
from ray.tune.utils import _detect_checkpoint_function
from ray.util.annotations import PublicAPI
class ResourceTrainable(trainable):

    @classmethod
    def default_resource_request(cls, config: Dict[str, Any]) -> Optional[PlacementGroupFactory]:
        if not isinstance(pgf, PlacementGroupFactory) and callable(pgf):
            return pgf(config)
        return pgf
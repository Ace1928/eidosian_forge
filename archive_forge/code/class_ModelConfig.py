import abc
from dataclasses import dataclass, field
import functools
from typing import Callable, List, Optional, Tuple, TYPE_CHECKING, Union
import numpy as np
from ray.rllib.models.torch.misc import (
from ray.rllib.models.utils import get_activation_fn
from ray.rllib.utils.annotations import ExperimentalAPI
@ExperimentalAPI
@dataclass
class ModelConfig(abc.ABC):
    """Base class for configuring a `Model` instance.

    ModelConfigs are DL framework-agnostic.
    A `Model` (as a sub-component of an `RLModule`) is built via calling the
    respective ModelConfig's `build()` method.
    RLModules build their sub-components this way after receiving one or more
    `ModelConfig` instances from a Catalog object.

    However, `ModelConfig` is not restricted to be used only with Catalog or RLModules.
    Usage examples can be found in the individual Model classes', e.g.
    see `ray.rllib.core.models.configs::MLPHeadConfig`.

    Attributes:
        input_dims: The input dimensions of the network
        always_check_shapes: Whether to always check the inputs and outputs of the
            model for the specifications. Input specifications are checked on failed
            forward passes of the model regardless of this flag. If this flag is set
            to `True`, inputs and outputs are checked on every call. This leads to
            a slow-down and should only be used for debugging.
    """
    input_dims: Union[List[int], Tuple[int]] = None
    always_check_shapes: bool = False

    @abc.abstractmethod
    def build(self, framework: str):
        """Builds the model.

        Args:
            framework: The framework to use for building the model.
        """
        raise NotImplementedError

    @property
    def output_dims(self) -> Optional[Tuple[int]]:
        """Read-only `output_dims` are inferred automatically from other settings."""
        return None
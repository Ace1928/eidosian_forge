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
class MLPHeadConfig(_MLPConfig):
    """Configuration for an MLP head.

    See _MLPConfig for usage details.

    Example:

    .. testcode::

        # Configuration:
        config = MLPHeadConfig(
            input_dims=[4],  # must be 1D tensor
            hidden_layer_dims=[8, 8],
            hidden_layer_activation="relu",
            hidden_layer_use_layernorm=False,
            # final output layer with no activation (linear)
            output_layer_dim=2,
            output_layer_activation="linear",
        )
        model = config.build(framework="tf2")

        # Resulting stack in pseudocode:
        # Linear(4, 8, bias=True)
        # ReLU()
        # Linear(8, 8, bias=True)
        # ReLU()
        # Linear(8, 2, bias=True)

    Example:

    .. testcode::

        # Configuration:
        config = MLPHeadConfig(
            input_dims=[2],
            hidden_layer_dims=[10, 4],
            hidden_layer_activation="silu",
            hidden_layer_use_layernorm=True,
            hidden_layer_use_bias=False,
            # No final output layer (use last dim in `hidden_layer_dims`
            # as the size of the last layer in the stack).
            output_layer_dim=None,
        )
        model = config.build(framework="torch")

        # Resulting stack in pseudocode:
        # Linear(2, 10, bias=False)
        # LayerNorm((10,))  # layer norm always before activation
        # SiLU()
        # Linear(10, 4, bias=False)
        # LayerNorm((4,))  # layer norm always before activation
        # SiLU()
    """

    @_framework_implemented()
    def build(self, framework: str='torch') -> 'Model':
        self._validate(framework=framework)
        if framework == 'torch':
            from ray.rllib.core.models.torch.heads import TorchMLPHead
            return TorchMLPHead(self)
        else:
            from ray.rllib.core.models.tf.heads import TfMLPHead
            return TfMLPHead(self)
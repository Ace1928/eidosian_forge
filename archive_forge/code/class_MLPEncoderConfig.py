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
class MLPEncoderConfig(_MLPConfig):
    """Configuration for an MLP that acts as an encoder.

    See _MLPConfig for usage details.

    Example:
    .. testcode::

        # Configuration:
        config = MLPEncoderConfig(
            input_dims=[4],  # must be 1D tensor
            hidden_layer_dims=[16],
            hidden_layer_activation="relu",
            hidden_layer_use_layernorm=False,
            output_layer_dim=None,  # maybe None or an int
        )
        model = config.build(framework="torch")

        # Resulting stack in pseudocode:
        # Linear(4, 16, bias=True)
        # ReLU()

    Example:
    .. testcode::

        # Configuration:
        config = MLPEncoderConfig(
            input_dims=[2],
            hidden_layer_dims=[8, 8],
            hidden_layer_activation="silu",
            hidden_layer_use_layernorm=True,
            hidden_layer_use_bias=False,
            output_layer_dim=4,
            output_layer_activation="tanh",
            output_layer_use_bias=False,
        )
        model = config.build(framework="tf2")

        # Resulting stack in pseudocode:
        # Linear(2, 8, bias=False)
        # LayerNorm((8,))  # layernorm always before activation
        # SiLU()
        # Linear(8, 8, bias=False)
        # LayerNorm((8,))  # layernorm always before activation
        # SiLU()
        # Linear(8, 4, bias=False)
        # Tanh()
    """

    @_framework_implemented()
    def build(self, framework: str='torch') -> 'Encoder':
        self._validate(framework)
        if framework == 'torch':
            from ray.rllib.core.models.torch.encoder import TorchMLPEncoder
            return TorchMLPEncoder(self)
        else:
            from ray.rllib.core.models.tf.encoder import TfMLPEncoder
            return TfMLPEncoder(self)
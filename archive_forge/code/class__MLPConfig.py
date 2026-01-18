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
class _MLPConfig(ModelConfig):
    """Generic configuration class for multi-layer-perceptron based Model classes.

    `output_dims` is reached by either the provided `output_layer_dim` setting (int) OR
    by the last entry of `hidden_layer_dims`. In the latter case, no special output
    layer is added and all layers in the stack behave exactly the same. If
    `output_layer_dim` is provided, users might also change this last layer's
    activation (`output_layer_activation`) and its bias setting
    (`output_layer_use_bias`).

    This is a private class as users should not configure their models directly
    through this class, but use one of the sub-classes, e.g. `MLPHeadConfig` or
    `MLPEncoderConfig`.

    Attributes:
        input_dims: A 1D tensor indicating the input dimension, e.g. `[32]`.
        hidden_layer_dims: The sizes of the hidden layers. If an empty list,
            `output_layer_dim` must be provided (int) and only a single layer will be
            built.
        hidden_layer_use_bias: Whether to use bias on all dense layers in the network
            (excluding a possible separate output layer defined by `output_layer_dim`).
        hidden_layer_activation: The activation function to use after each layer (
            except for the output). The default activation for hidden layers is "relu".
        hidden_layer_use_layernorm: Whether to insert a LayerNorm functionality
            in between each hidden layer's output and its activation.
        output_layer_dim: An int indicating the size of the output layer. This may be
            set to `None` in case no extra output layer should be built and only the
            layers specified by `hidden_layer_dims` will be part of the network.
        output_layer_use_bias: Whether to use bias on the separate output layer, if any.
        output_layer_activation: The activation function to use for the output layer,
            if any. The default activation for the output layer, if any, is "linear",
            meaning no activation.
    """
    hidden_layer_dims: Union[List[int], Tuple[int]] = (256, 256)
    hidden_layer_use_bias: bool = True
    hidden_layer_activation: str = 'relu'
    hidden_layer_use_layernorm: bool = False
    output_layer_dim: Optional[int] = None
    output_layer_use_bias: bool = True
    output_layer_activation: str = 'linear'

    @property
    def output_dims(self):
        if self.output_layer_dim is None and (not self.hidden_layer_dims):
            raise ValueError('If `output_layer_dim` is None, you must specify at least one hidden layer dim, e.g. `hidden_layer_dims=[32]`!')
        return (self.output_layer_dim or self.hidden_layer_dims[-1],)

    def _validate(self, framework: str='torch'):
        """Makes sure that settings are valid."""
        if self.input_dims is not None and len(self.input_dims) != 1:
            raise ValueError(f'`input_dims` ({self.input_dims}) of MLPConfig must be 1D, e.g. `[32]`!')
        if len(self.output_dims) != 1:
            raise ValueError(f'`output_dims` ({self.output_dims}) of _MLPConfig must be 1D, e.g. `[32]`! This is an inferred value, hence other settings might be wrong.')
        get_activation_fn(self.hidden_layer_activation, framework=framework)
        get_activation_fn(self.output_layer_activation, framework=framework)
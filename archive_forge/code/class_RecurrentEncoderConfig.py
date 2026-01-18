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
class RecurrentEncoderConfig(ModelConfig):
    """Configuration for an LSTM-based or a GRU-based encoder.

    The encoder consists of...
    - Zero or one tokenizers
    - N LSTM/GRU layers stacked on top of each other and feeding
    their outputs as inputs to the respective next layer.
    - One linear output layer

    This makes for the following flow of tensors:

    Inputs
    |
    [Tokenizer if present]
    |
    LSTM layer 1
    |
    (...)
    |
    LSTM layer n
    |
    Linear output layer
    |
    Outputs

    The internal state is structued as (num_layers, B, hidden-size) for all hidden
    state components, e.g.
    h- and c-states of the LSTM layer(s) or h-state of the GRU layer(s).
    For example, the hidden states of an LSTMEncoder with num_layers=2 and hidden_dim=8
    would be: {"h": (2, B, 8), "c": (2, B, 8)}.

    `output_dims` is reached by the last recurrent layer's dimension, which is always
    the `hidden_dims` value.

    Example:
    .. testcode::

        # Configuration:
        config = RecurrentEncoderConfig(
            recurrent_layer_type="lstm",
            input_dims=[16],  # must be 1D tensor
            hidden_dim=128,
            num_layers=2,
            use_bias=True,
        )
        model = config.build(framework="torch")

        # Resulting stack in pseudocode:
        # LSTM(16, 128, bias=True)
        # LSTM(128, 128, bias=True)

        # Resulting shape of the internal states (c- and h-states):
        # (2, B, 128) for each c- and h-states.

    Example:
    .. testcode::

        # Configuration:
        config = RecurrentEncoderConfig(
            recurrent_layer_type="gru",
            input_dims=[32],  # must be 1D tensor
            hidden_dim=64,
            num_layers=1,
            use_bias=False,
        )
        model = config.build(framework="torch")

        # Resulting stack in pseudocode:
        # GRU(32, 64, bias=False)

        # Resulting shape of the internal state:
        # (1, B, 64)

    Attributes:
        input_dims: The input dimensions. Must be 1D. This is the 1D shape of the tensor
            that goes into the first recurrent layer.
        recurrent_layer_type: The type of the recurrent layer(s).
            Either "lstm" or "gru".
        hidden_dim: The size of the hidden internal state(s) of the recurrent layer(s).
            For example, for an LSTM, this would be the size of the c- and h-tensors.
        num_layers: The number of recurrent (LSTM or GRU) layers to stack.
        batch_major: Wether the input is batch major (B, T, ..) or
            time major (T, B, ..).
        use_bias: Whether to use bias on the recurrent layers in the network.
        view_requirements_dict: The view requirements to use if anything else than
            observation_space or action_space is to be encoded. This signifies an
            advanced use case.
        tokenizer_config: A ModelConfig to build tokenizers for observations,
            actions and other spaces that might be present in the
            view_requirements_dict.
    """
    recurrent_layer_type: str = 'lstm'
    hidden_dim: int = None
    num_layers: int = None
    batch_major: bool = True
    use_bias: bool = True
    tokenizer_config: ModelConfig = None

    @property
    def output_dims(self):
        return (self.hidden_dim,)

    def _validate(self, framework: str='torch'):
        """Makes sure that settings are valid."""
        if self.recurrent_layer_type not in ['gru', 'lstm']:
            raise ValueError(f"`recurrent_layer_type` ({self.recurrent_layer_type}) of RecurrentEncoderConfig must be 'gru' or 'lstm'!")
        if self.input_dims is not None and len(self.input_dims) != 1:
            raise ValueError(f'`input_dims` ({self.input_dims}) of RecurrentEncoderConfig must be 1D, e.g. `[32]`!')
        if len(self.output_dims) != 1:
            raise ValueError(f'`output_dims` ({self.output_dims}) of RecurrentEncoderConfig must be 1D, e.g. `[32]`! This is an inferred value, hence other settings might be wrong.')

    @_framework_implemented()
    def build(self, framework: str='torch') -> 'Encoder':
        if framework == 'torch':
            from ray.rllib.core.models.torch.encoder import TorchGRUEncoder as GRU, TorchLSTMEncoder as LSTM
        else:
            from ray.rllib.core.models.tf.encoder import TfGRUEncoder as GRU, TfLSTMEncoder as LSTM
        if self.recurrent_layer_type == 'lstm':
            return LSTM(self)
        else:
            return GRU(self)
from __future__ import annotations
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple
import tensorflow as tf
from .utils import ModelOutput
@dataclass
class TFBaseModelOutputWithNoAttention(ModelOutput):
    """
    Base class for model's outputs, with potential hidden states.

    Args:
        last_hidden_state (`tf.Tensor` shape `(batch_size, num_channels, height, width)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings, if the model has an embedding layer, + one for
            the output of each layer) of shape `(batch_size, num_channels, height, width)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    """
    last_hidden_state: tf.Tensor = None
    hidden_states: Optional[Tuple[tf.Tensor, ...]] = None
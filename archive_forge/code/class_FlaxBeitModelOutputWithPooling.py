from typing import Callable, List, Optional, Tuple
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from ...modeling_flax_outputs import (
from ...modeling_flax_utils import (
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward
from .configuration_beit import BeitConfig
@flax.struct.dataclass
class FlaxBeitModelOutputWithPooling(FlaxBaseModelOutputWithPooling):
    """
    Class for outputs of [`FlaxBeitModel`].

    Args:
        last_hidden_state (`jnp.ndarray` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (`jnp.ndarray` of shape `(batch_size, hidden_size)`):
            Average of the last layer hidden states of the patch tokens (excluding the *[CLS]* token) if
            *config.use_mean_pooling* is set to True. If set to False, then the final hidden state of the *[CLS]* token
            will be returned.
        hidden_states (`tuple(jnp.ndarray)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `jnp.ndarray` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer plus
            the initial embedding outputs.
        attentions (`tuple(jnp.ndarray)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """
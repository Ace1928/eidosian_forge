import copy
from typing import Callable, Optional, Tuple
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from flax.linen import partitioning as nn_partitioning
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax.random import PRNGKey
from ...modeling_flax_outputs import (
from ...modeling_flax_utils import (
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_t5 import T5Config
class FlaxT5LayerCrossAttention(nn.Module):
    config: T5Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.EncDecAttention = FlaxT5Attention(self.config, has_relative_attention_bias=False, causal=False, dtype=self.dtype)
        self.layer_norm = FlaxT5LayerNorm(self.config.d_model, eps=self.config.layer_norm_epsilon, dtype=self.dtype)
        self.dropout = nn.Dropout(self.config.dropout_rate)

    def __call__(self, hidden_states, key_value_states, attention_mask=None, position_bias=None, output_attentions=False, deterministic=True):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.EncDecAttention(normed_hidden_states, attention_mask=attention_mask, key_value_states=key_value_states, position_bias=position_bias, output_attentions=output_attentions)
        hidden_states = hidden_states + self.dropout(attention_output[0], deterministic=deterministic)
        outputs = (hidden_states,) + attention_output[1:]
        return outputs
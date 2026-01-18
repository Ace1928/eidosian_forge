import copy
from typing import Any, Callable, List, Optional, Tuple
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
from .configuration_longt5 import LongT5Config
class FlaxLongT5LayerTransientGlobalSelfAttention(nn.Module):
    """Transient-Global self attention used in encoder"""
    config: LongT5Config
    has_relative_attention_bias: bool = False
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.TransientGlobalSelfAttention = FlaxLongT5TransientGlobalAttention(self.config, has_relative_attention_bias=self.has_relative_attention_bias, dtype=self.dtype)
        self.layer_norm = FlaxLongT5LayerNorm(self.config.d_model, eps=self.config.layer_norm_epsilon, dtype=self.dtype)
        self.dropout = nn.Dropout(self.config.dropout_rate)

    def __call__(self, hidden_states, attention_mask=None, position_bias=None, output_attentions=False, deterministic=True, **kwargs: Any):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.TransientGlobalSelfAttention(normed_hidden_states, attention_mask=attention_mask, position_bias=position_bias, output_attentions=output_attentions, deterministic=deterministic)
        hidden_states = hidden_states + self.dropout(attention_output[0], deterministic=deterministic)
        outputs = (hidden_states,) + attention_output[1:]
        return outputs
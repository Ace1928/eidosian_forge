from typing import Callable, Optional, Tuple
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from ...modeling_flax_outputs import (
from ...modeling_flax_utils import (
from ...utils import ModelOutput, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_albert import AlbertConfig
class FlaxAlbertLayer(nn.Module):
    config: AlbertConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.attention = FlaxAlbertSelfAttention(self.config, dtype=self.dtype)
        self.ffn = nn.Dense(self.config.intermediate_size, kernel_init=jax.nn.initializers.normal(self.config.initializer_range), dtype=self.dtype)
        self.activation = ACT2FN[self.config.hidden_act]
        self.ffn_output = nn.Dense(self.config.hidden_size, kernel_init=jax.nn.initializers.normal(self.config.initializer_range), dtype=self.dtype)
        self.full_layer_layer_norm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)

    def __call__(self, hidden_states, attention_mask, deterministic: bool=True, output_attentions: bool=False):
        attention_outputs = self.attention(hidden_states, attention_mask, deterministic=deterministic, output_attentions=output_attentions)
        attention_output = attention_outputs[0]
        ffn_output = self.ffn(attention_output)
        ffn_output = self.activation(ffn_output)
        ffn_output = self.ffn_output(ffn_output)
        ffn_output = self.dropout(ffn_output, deterministic=deterministic)
        hidden_states = self.full_layer_layer_norm(ffn_output + attention_output)
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attention_outputs[1],)
        return outputs
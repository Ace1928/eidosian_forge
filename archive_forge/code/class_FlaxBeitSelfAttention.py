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
class FlaxBeitSelfAttention(nn.Module):
    config: BeitConfig
    window_size: Tuple[int, int]
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if self.config.hidden_size % self.config.num_attention_heads != 0 and (not hasattr(self.config, 'embedding_size')):
            raise ValueError(f'The hidden size {(self.config.hidden_size,)} is not a multiple of the number of attention heads {self.config.num_attention_heads}.')
        self.query = nn.Dense(self.config.hidden_size, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(self.config.initializer_range))
        self.key = nn.Dense(self.config.hidden_size, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(self.config.initializer_range), use_bias=False)
        self.value = nn.Dense(self.config.hidden_size, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(self.config.initializer_range))
        self.relative_position_bias = FlaxBeitRelativePositionBias(self.config, window_size=self.window_size, dtype=self.dtype) if self.window_size else None

    def __call__(self, hidden_states, relative_position_bias=None, deterministic: bool=True, output_attentions: bool=False):
        head_dim = self.config.hidden_size // self.config.num_attention_heads
        query_states = self.query(hidden_states).reshape(hidden_states.shape[:2] + (self.config.num_attention_heads, head_dim))
        value_states = self.value(hidden_states).reshape(hidden_states.shape[:2] + (self.config.num_attention_heads, head_dim))
        key_states = self.key(hidden_states).reshape(hidden_states.shape[:2] + (self.config.num_attention_heads, head_dim))
        dropout_rng = None
        if not deterministic and self.config.attention_probs_dropout_prob > 0.0:
            dropout_rng = self.make_rng('dropout')
        attention_bias = jnp.array(0.0, dtype=self.dtype)
        if self.relative_position_bias is not None:
            attention_bias = jnp.expand_dims(self.relative_position_bias(), 0)
            attention_bias = attention_bias.astype(query_states.dtype)
        if relative_position_bias is not None:
            attention_bias = attention_bias + relative_position_bias.astype(attention_bias.dtype)
        attn_weights = dot_product_attention_weights(query_states, key_states, bias=attention_bias, dropout_rng=dropout_rng, dropout_rate=self.config.attention_probs_dropout_prob, broadcast_dropout=True, deterministic=deterministic, dtype=self.dtype, precision=None)
        attn_output = jnp.einsum('...hqk,...khd->...qhd', attn_weights, value_states)
        attn_output = attn_output.reshape(attn_output.shape[:2] + (-1,))
        outputs = (attn_output, attn_weights) if output_attentions else (attn_output,)
        return outputs
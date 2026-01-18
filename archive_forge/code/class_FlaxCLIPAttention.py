from typing import Any, Optional, Tuple, Union
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from ...modeling_flax_outputs import FlaxBaseModelOutput, FlaxBaseModelOutputWithPooling
from ...modeling_flax_utils import (
from ...utils import ModelOutput, add_start_docstrings, logging
from .configuration_clip import CLIPConfig, CLIPTextConfig, CLIPVisionConfig
class FlaxCLIPAttention(nn.Module):
    config: Union[CLIPTextConfig, CLIPVisionConfig]
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.embed_dim = self.config.hidden_size
        self.num_heads = self.config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(f'embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads}).')
        self.scale = self.head_dim ** (-0.5)
        self.dropout = self.config.attention_dropout
        self.k_proj = nn.Dense(self.embed_dim, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(0.01))
        self.v_proj = nn.Dense(self.embed_dim, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(0.01))
        self.q_proj = nn.Dense(self.embed_dim, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(0.01))
        self.out_proj = nn.Dense(self.embed_dim, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(0.01))
        self.causal = isinstance(self.config, CLIPTextConfig)
        if self.causal:
            self.causal_mask = make_causal_mask(jnp.ones((1, self.config.max_position_embeddings), dtype='i4'))

    def _split_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.num_heads, self.head_dim))

    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.embed_dim,))

    def __call__(self, hidden_states, attention_mask=None, deterministic: bool=True, output_attentions: bool=False):
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)
        causal_attention_mask = None
        if self.causal:
            query_length, key_length = (query.shape[1], key.shape[1])
            causal_attention_mask = self.causal_mask[:, :, key_length - query_length:key_length, :key_length]
        if attention_mask is not None and causal_attention_mask is not None:
            attention_mask = jnp.expand_dims(attention_mask, axis=(-3, -2))
            attention_mask = combine_masks(attention_mask, causal_attention_mask, dtype='i4')
        elif causal_attention_mask is not None:
            attention_mask = causal_attention_mask
        elif attention_mask is not None:
            attention_mask = jnp.expand_dims(attention_mask, axis=(-3, -2))
        if attention_mask is not None:
            attention_bias = lax.select(attention_mask > 0, jnp.full(attention_mask.shape, 0.0).astype(self.dtype), jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(self.dtype))
        else:
            attention_bias = None
        dropout_rng = None
        if not deterministic and self.dropout > 0.0:
            dropout_rng = self.make_rng('dropout')
        attn_weights = dot_product_attention_weights(query, key, bias=attention_bias, dropout_rng=dropout_rng, dropout_rate=self.dropout, deterministic=deterministic, dtype=self.dtype, precision=None)
        attn_output = jnp.einsum('...hqk,...khd->...qhd', attn_weights, value)
        attn_output = self._merge_heads(attn_output)
        attn_output = self.out_proj(attn_output)
        outputs = (attn_output, attn_weights) if output_attentions else (attn_output,)
        return outputs
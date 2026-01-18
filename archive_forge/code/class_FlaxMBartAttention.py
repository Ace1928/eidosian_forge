import math
import random
from functools import partial
from typing import Callable, Optional, Tuple
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from jax.random import PRNGKey
from ...modeling_flax_outputs import (
from ...modeling_flax_utils import (
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_mbart import MBartConfig
class FlaxMBartAttention(nn.Module):
    config: MBartConfig
    embed_dim: int
    num_heads: int
    dropout: float = 0.0
    causal: bool = False
    bias: bool = True
    dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(f'embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads}).')
        dense = partial(nn.Dense, self.embed_dim, use_bias=self.bias, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(self.config.init_std))
        self.q_proj, self.k_proj, self.v_proj = (dense(), dense(), dense())
        self.out_proj = dense()
        self.dropout_layer = nn.Dropout(rate=self.dropout)
        if self.causal:
            self.causal_mask = make_causal_mask(jnp.ones((1, self.config.max_position_embeddings), dtype='bool'), dtype='bool')

    def _split_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.num_heads, self.head_dim))

    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.embed_dim,))

    @nn.compact
    def _concatenate_to_cache(self, key, value, query, attention_mask):
        """
        This function takes projected key, value states from a single input token and concatenates the states to cached
        states from previous steps. This function is slighly adapted from the official Flax repository:
        https://github.com/google/flax/blob/491ce18759622506588784b4fca0e4bf05f8c8cd/flax/linen/attention.py#L252
        """
        is_initialized = self.has_variable('cache', 'cached_key')
        cached_key = self.variable('cache', 'cached_key', jnp.zeros, key.shape, key.dtype)
        cached_value = self.variable('cache', 'cached_value', jnp.zeros, value.shape, value.dtype)
        cache_index = self.variable('cache', 'cache_index', lambda: jnp.array(0, dtype=jnp.int32))
        if is_initialized:
            *batch_dims, max_length, num_heads, depth_per_head = cached_key.value.shape
            cur_index = cache_index.value
            indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
            key = lax.dynamic_update_slice(cached_key.value, key, indices)
            value = lax.dynamic_update_slice(cached_value.value, value, indices)
            cached_key.value = key
            cached_value.value = value
            num_updated_cache_vectors = query.shape[1]
            cache_index.value = cache_index.value + num_updated_cache_vectors
            pad_mask = jnp.broadcast_to(jnp.arange(max_length) < cur_index + num_updated_cache_vectors, tuple(batch_dims) + (1, num_updated_cache_vectors, max_length))
            attention_mask = combine_masks(pad_mask, attention_mask)
        return (key, value, attention_mask)

    def __call__(self, hidden_states: jnp.ndarray, key_value_states: Optional[jnp.ndarray]=None, attention_mask: Optional[jnp.ndarray]=None, init_cache: bool=False, deterministic: bool=True) -> Tuple[jnp.ndarray]:
        """Input shape: Batch x Time x Channel"""
        is_cross_attention = key_value_states is not None
        batch_size = hidden_states.shape[0]
        query_states = self.q_proj(hidden_states)
        if is_cross_attention:
            key_states = self.k_proj(key_value_states)
            value_states = self.v_proj(key_value_states)
        else:
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
        query_states = self._split_heads(query_states)
        key_states = self._split_heads(key_states)
        value_states = self._split_heads(value_states)
        if self.causal:
            query_length, key_length = (query_states.shape[1], key_states.shape[1])
            if self.has_variable('cache', 'cached_key'):
                mask_shift = self.variables['cache']['cache_index']
                max_decoder_length = self.variables['cache']['cached_key'].shape[1]
                causal_mask = lax.dynamic_slice(self.causal_mask, (0, 0, mask_shift, 0), (1, 1, query_length, max_decoder_length))
            else:
                causal_mask = self.causal_mask[:, :, :query_length, :key_length]
            causal_mask = jnp.broadcast_to(causal_mask, (batch_size,) + causal_mask.shape[1:])
        if attention_mask is not None and self.causal:
            attention_mask = jnp.broadcast_to(jnp.expand_dims(attention_mask, axis=(-3, -2)), causal_mask.shape)
            attention_mask = combine_masks(attention_mask, causal_mask)
        elif self.causal:
            attention_mask = causal_mask
        elif attention_mask is not None:
            attention_mask = jnp.expand_dims(attention_mask, axis=(-3, -2))
        if self.causal and (self.has_variable('cache', 'cached_key') or init_cache):
            key_states, value_states, attention_mask = self._concatenate_to_cache(key_states, value_states, query_states, attention_mask)
        if attention_mask is not None:
            attention_bias = lax.select(attention_mask > 0, jnp.full(attention_mask.shape, 0.0).astype(self.dtype), jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(self.dtype))
        else:
            attention_bias = None
        dropout_rng = None
        if not deterministic and self.dropout > 0.0:
            dropout_rng = self.make_rng('dropout')
        attn_weights = dot_product_attention_weights(query_states, key_states, bias=attention_bias, dropout_rng=dropout_rng, dropout_rate=self.dropout, broadcast_dropout=True, deterministic=deterministic, dtype=self.dtype, precision=None)
        attn_output = jnp.einsum('...hqk,...khd->...qhd', attn_weights, value_states)
        attn_output = self._merge_heads(attn_output)
        attn_output = self.out_proj(attn_output)
        return (attn_output, attn_weights)
from functools import partial
from typing import Optional, Tuple, Union
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from ...modeling_flax_outputs import FlaxBaseModelOutput, FlaxCausalLMOutput
from ...modeling_flax_utils import (
from ...utils import ModelOutput, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_wav2vec2 import Wav2Vec2Config
class FlaxWav2Vec2Attention(nn.Module):
    config: Wav2Vec2Config
    embed_dim: int
    num_heads: int
    dropout: float = 0.0
    bias: bool = True
    dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(f'embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads}).')
        dense = partial(nn.Dense, self.embed_dim, use_bias=self.bias, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(self.config.initializer_range))
        self.q_proj, self.k_proj, self.v_proj = (dense(), dense(), dense())
        self.out_proj = dense()
        self.dropout_layer = nn.Dropout(rate=self.dropout)

    def _split_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.num_heads, self.head_dim))

    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.embed_dim,))

    def __call__(self, hidden_states: jnp.ndarray, key_value_states: Optional[jnp.ndarray]=None, attention_mask: Optional[jnp.ndarray]=None, deterministic: bool=True) -> Tuple[jnp.ndarray]:
        """Input shape: Batch x Time x Channel"""
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        query_states = self._split_heads(query_states)
        key_states = self._split_heads(key_states)
        value_states = self._split_heads(value_states)
        if attention_mask is not None:
            attention_mask = jnp.expand_dims(attention_mask, axis=(-3, -2))
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
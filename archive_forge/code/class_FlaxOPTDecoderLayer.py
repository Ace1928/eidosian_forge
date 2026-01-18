from functools import partial
from typing import Optional, Tuple
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from jax.random import PRNGKey
from ...modeling_flax_outputs import FlaxBaseModelOutput, FlaxMaskedLMOutput
from ...modeling_flax_utils import ACT2FN, FlaxPreTrainedModel, append_call_sample_docstring
from ...utils import add_start_docstrings, logging
from .configuration_opt import OPTConfig
class FlaxOPTDecoderLayer(nn.Module):
    config: OPTConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        self.embed_dim = self.config.hidden_size
        self.self_attn = FlaxOPTAttention(config=self.config, embed_dim=self.embed_dim, num_heads=self.config.num_attention_heads, dropout=self.config.attention_dropout, causal=True, dtype=self.dtype)
        self.do_layer_norm_before = self.config.do_layer_norm_before
        self.dropout_layer = nn.Dropout(rate=self.config.dropout)
        self.activation_fn = ACT2FN[self.config.activation_function]
        self.self_attn_layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)
        self.fc1 = nn.Dense(self.config.ffn_dim, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(self.config.init_std))
        self.fc2 = nn.Dense(self.embed_dim, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(self.config.init_std))
        self.final_layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)

    def __call__(self, hidden_states: jnp.ndarray, attention_mask: jnp.ndarray, init_cache: bool=False, output_attentions: bool=True, deterministic: bool=True) -> Tuple[jnp.ndarray]:
        residual = hidden_states
        if self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, self_attn_weights = self.self_attn(hidden_states=hidden_states, attention_mask=attention_mask, init_cache=init_cache, deterministic=deterministic)
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)
        hidden_states = residual + hidden_states
        if not self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        residual = hidden_states
        if self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)
        hidden_states = (residual + hidden_states).reshape(hidden_states_shape)
        if not self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        return outputs
from typing import Any, Optional, Tuple
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from ...modeling_flax_outputs import (
from ...modeling_flax_utils import ACT2FN, FlaxPreTrainedModel, append_call_sample_docstring
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_gpt2 import GPT2Config
class FlaxGPT2Block(nn.Module):
    config: GPT2Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        hidden_size = self.config.hidden_size
        inner_dim = self.config.n_inner if self.config.n_inner is not None else 4 * hidden_size
        self.ln_1 = nn.LayerNorm(epsilon=self.config.layer_norm_epsilon, dtype=self.dtype)
        self.attn = FlaxGPT2Attention(self.config, dtype=self.dtype)
        self.ln_2 = nn.LayerNorm(epsilon=self.config.layer_norm_epsilon, dtype=self.dtype)
        if self.config.add_cross_attention:
            self.crossattention = FlaxGPT2Attention(config=self.config, dtype=self.dtype, causal=False, is_cross_attention=True)
            self.ln_cross_attn = nn.LayerNorm(epsilon=self.config.layer_norm_epsilon, dtype=self.dtype)
        self.mlp = FlaxGPT2MLP(self.config, inner_dim, dtype=self.dtype)

    def __call__(self, hidden_states, attention_mask=None, encoder_hidden_states: Optional[jnp.ndarray]=None, encoder_attention_mask: Optional[jnp.ndarray]=None, deterministic: bool=True, init_cache: bool=False, output_attentions: bool=False):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(hidden_states, attention_mask=attention_mask, deterministic=deterministic, init_cache=init_cache, output_attentions=output_attentions)
        attn_output = attn_outputs[0]
        outputs = attn_outputs[1:]
        hidden_states = attn_output + residual
        if encoder_hidden_states is not None:
            if not hasattr(self, 'crossattention'):
                raise ValueError(f'If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`')
            residual = hidden_states
            hidden_states = self.ln_cross_attn(hidden_states)
            cross_attn_outputs = self.crossattention(hidden_states, key_value_states=encoder_hidden_states, attention_mask=encoder_attention_mask, deterministic=deterministic, output_attentions=output_attentions)
            attn_output = cross_attn_outputs[0]
            hidden_states = residual + attn_output
            outputs = outputs + cross_attn_outputs[1:]
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states, deterministic=deterministic)
        hidden_states = residual + feed_forward_hidden_states
        outputs = (hidden_states,) + outputs
        return outputs
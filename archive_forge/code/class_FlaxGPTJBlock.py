from functools import partial
from typing import Optional, Tuple
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from ...modeling_flax_outputs import FlaxBaseModelOutput, FlaxCausalLMOutput
from ...modeling_flax_utils import ACT2FN, FlaxPreTrainedModel, append_call_sample_docstring
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_gptj import GPTJConfig
class FlaxGPTJBlock(nn.Module):
    config: GPTJConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        hidden_size = self.config.hidden_size
        inner_dim = self.config.n_inner if self.config.n_inner is not None else 4 * hidden_size
        self.ln_1 = nn.LayerNorm(epsilon=self.config.layer_norm_epsilon, dtype=self.dtype)
        self.attn = FlaxGPTJAttention(self.config, dtype=self.dtype)
        self.mlp = FlaxGPTJMLP(self.config, inner_dim, dtype=self.dtype)

    def __call__(self, hidden_states, attention_mask=None, position_ids=None, deterministic: bool=True, init_cache: bool=False, output_attentions: bool=False):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(hidden_states, attention_mask=attention_mask, position_ids=position_ids, deterministic=deterministic, init_cache=init_cache, output_attentions=output_attentions)
        attn_output = attn_outputs[0]
        feed_forward_hidden_states = self.mlp(hidden_states, deterministic=deterministic)
        hidden_states = attn_output + feed_forward_hidden_states + residual
        return (hidden_states,) + attn_outputs[1:]
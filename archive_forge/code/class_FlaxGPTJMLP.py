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
class FlaxGPTJMLP(nn.Module):
    config: GPTJConfig
    intermediate_size: int
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        embed_dim = self.config.hidden_size
        kernel_init = jax.nn.initializers.normal(self.config.initializer_range)
        self.fc_in = nn.Dense(self.intermediate_size, dtype=self.dtype, kernel_init=kernel_init)
        self.fc_out = nn.Dense(embed_dim, dtype=self.dtype, kernel_init=kernel_init)
        self.act = ACT2FN[self.config.activation_function]
        self.dropout = nn.Dropout(rate=self.config.resid_pdrop)

    def __call__(self, hidden_states, deterministic: bool=True):
        hidden_states = self.fc_in(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc_out(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        return hidden_states
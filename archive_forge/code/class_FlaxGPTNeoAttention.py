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
from ...modeling_flax_outputs import FlaxBaseModelOutput, FlaxCausalLMOutput
from ...modeling_flax_utils import ACT2FN, FlaxPreTrainedModel, append_call_sample_docstring
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_gpt_neo import GPTNeoConfig
class FlaxGPTNeoAttention(nn.Module):
    config: GPTNeoConfig
    layer_id: int = 0
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        attention_type = self.config.attention_layers[self.layer_id]
        self.attention = FlaxGPTNeoSelfAttention(self.config, attention_type, dtype=self.dtype)

    def __call__(self, hidden_states, attention_mask=None, deterministic: bool=True, init_cache: bool=False, output_attentions: bool=False):
        return self.attention(hidden_states, attention_mask=attention_mask, deterministic=deterministic, init_cache=init_cache, output_attentions=output_attentions)
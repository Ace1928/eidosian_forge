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
class FlaxCLIPEncoderLayer(nn.Module):
    config: Union[CLIPTextConfig, CLIPVisionConfig]
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.self_attn = FlaxCLIPAttention(self.config, dtype=self.dtype)
        self.layer_norm1 = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        self.mlp = FlaxCLIPMLP(self.config, dtype=self.dtype)
        self.layer_norm2 = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)

    def __call__(self, hidden_states, attention_mask, deterministic: bool=True, output_attentions: bool=False):
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        attn_outputs = self.self_attn(hidden_states=hidden_states, attention_mask=attention_mask, deterministic=deterministic, output_attentions=output_attentions)
        hidden_states = attn_outputs[0]
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        outputs = (hidden_states,)
        if output_attentions:
            outputs += attn_outputs[1:]
        return outputs
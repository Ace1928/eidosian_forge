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
class FlaxWav2Vec2EncoderLayerStableLayerNorm(nn.Module):
    config: Wav2Vec2Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.attention = FlaxWav2Vec2Attention(config=self.config, embed_dim=self.config.hidden_size, num_heads=self.config.num_attention_heads, dropout=self.config.attention_dropout, dtype=self.dtype)
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        self.feed_forward = FlaxWav2Vec2FeedForward(self.config, dtype=self.dtype)
        self.final_layer_norm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)

    def __call__(self, hidden_states, attention_mask=None, deterministic=True, output_attentions=False):
        attn_residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)
        hidden_states, attn_weights = self.attention(hidden_states, attention_mask=attention_mask, deterministic=deterministic)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        hidden_states = attn_residual + hidden_states
        hidden_states = hidden_states + self.feed_forward(self.final_layer_norm(hidden_states), deterministic=deterministic)
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs
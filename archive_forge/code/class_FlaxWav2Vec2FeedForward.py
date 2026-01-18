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
class FlaxWav2Vec2FeedForward(nn.Module):
    config: Wav2Vec2Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.intermediate_dropout = nn.Dropout(rate=self.config.activation_dropout)
        self.intermediate_dense = nn.Dense(self.config.intermediate_size, kernel_init=jax.nn.initializers.normal(self.config.initializer_range), dtype=self.dtype)
        if isinstance(self.config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[self.config.hidden_act]
        else:
            self.intermediate_act_fn = self.config.hidden_act
        self.output_dense = nn.Dense(self.config.hidden_size, kernel_init=jax.nn.initializers.normal(self.config.initializer_range), dtype=self.dtype)
        self.output_dropout = nn.Dropout(rate=self.config.hidden_dropout)

    def __call__(self, hidden_states, deterministic=True):
        hidden_states = self.intermediate_dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.intermediate_dropout(hidden_states, deterministic=deterministic)
        hidden_states = self.output_dense(hidden_states)
        hidden_states = self.output_dropout(hidden_states, deterministic=deterministic)
        return hidden_states
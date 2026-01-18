from typing import Optional, Tuple
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from ...modeling_flax_outputs import FlaxBaseModelOutput, FlaxBaseModelOutputWithPooling, FlaxSequenceClassifierOutput
from ...modeling_flax_utils import (
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward
from .configuration_vit import ViTConfig
class FlaxViTIntermediate(nn.Module):
    config: ViTConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.dense = nn.Dense(self.config.intermediate_size, kernel_init=jax.nn.initializers.variance_scaling(self.config.initializer_range ** 2, 'fan_in', 'truncated_normal'), dtype=self.dtype)
        self.activation = ACT2FN[self.config.hidden_act]

    def __call__(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states
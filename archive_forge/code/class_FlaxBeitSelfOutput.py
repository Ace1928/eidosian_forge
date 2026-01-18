from typing import Callable, List, Optional, Tuple
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from ...modeling_flax_outputs import (
from ...modeling_flax_utils import (
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward
from .configuration_beit import BeitConfig
class FlaxBeitSelfOutput(nn.Module):
    config: BeitConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.dense = nn.Dense(self.config.hidden_size, kernel_init=jax.nn.initializers.normal(self.config.initializer_range), dtype=self.dtype)
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)

    def __call__(self, hidden_states, deterministic: bool=True):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        return hidden_states
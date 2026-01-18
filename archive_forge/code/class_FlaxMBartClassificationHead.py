import math
import random
from functools import partial
from typing import Callable, Optional, Tuple
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from jax.random import PRNGKey
from ...modeling_flax_outputs import (
from ...modeling_flax_utils import (
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_mbart import MBartConfig
class FlaxMBartClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""
    config: MBartConfig
    inner_dim: int
    num_classes: int
    pooler_dropout: float
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.dense = nn.Dense(self.inner_dim, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(self.config.init_std))
        self.dropout = nn.Dropout(rate=self.pooler_dropout)
        self.out_proj = nn.Dense(self.num_classes, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(self.config.init_std))

    def __call__(self, hidden_states: jnp.ndarray, deterministic: bool):
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        hidden_states = self.dense(hidden_states)
        hidden_states = jnp.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states
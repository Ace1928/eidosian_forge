from functools import partial
from typing import Optional, Tuple
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict
from transformers import RegNetConfig
from transformers.modeling_flax_outputs import (
from transformers.modeling_flax_utils import (
from transformers.utils import (
class FlaxRegNetSELayer(nn.Module):
    """
    Squeeze and Excitation layer (SE) proposed in [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507).
    """
    in_channels: int
    reduced_channels: int
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.pooler = partial(nn.avg_pool, padding=((0, 0), (0, 0)))
        self.attention = FlaxRegNetSELayerCollection(self.in_channels, self.reduced_channels, dtype=self.dtype)

    def __call__(self, hidden_state: jnp.ndarray) -> jnp.ndarray:
        pooled = self.pooler(hidden_state, window_shape=(hidden_state.shape[1], hidden_state.shape[2]), strides=(hidden_state.shape[1], hidden_state.shape[2]))
        attention = self.attention(pooled)
        hidden_state = hidden_state * attention
        return hidden_state
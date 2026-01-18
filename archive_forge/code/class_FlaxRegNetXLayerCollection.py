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
class FlaxRegNetXLayerCollection(nn.Module):
    config: RegNetConfig
    out_channels: int
    stride: int = 1
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        groups = max(1, self.out_channels // self.config.groups_width)
        self.layer = [FlaxRegNetConvLayer(self.out_channels, kernel_size=1, activation=self.config.hidden_act, dtype=self.dtype, name='0'), FlaxRegNetConvLayer(self.out_channels, stride=self.stride, groups=groups, activation=self.config.hidden_act, dtype=self.dtype, name='1'), FlaxRegNetConvLayer(self.out_channels, kernel_size=1, activation=None, dtype=self.dtype, name='2')]

    def __call__(self, hidden_state: jnp.ndarray, deterministic: bool=True) -> jnp.ndarray:
        for layer in self.layer:
            hidden_state = layer(hidden_state, deterministic=deterministic)
        return hidden_state
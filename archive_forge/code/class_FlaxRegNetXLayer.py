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
class FlaxRegNetXLayer(nn.Module):
    """
    RegNet's layer composed by three `3x3` convolutions, same as a ResNet bottleneck layer with reduction = 1.
    """
    config: RegNetConfig
    in_channels: int
    out_channels: int
    stride: int = 1
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        should_apply_shortcut = self.in_channels != self.out_channels or self.stride != 1
        self.shortcut = FlaxRegNetShortCut(self.out_channels, stride=self.stride, dtype=self.dtype) if should_apply_shortcut else Identity()
        self.layer = FlaxRegNetXLayerCollection(self.config, in_channels=self.in_channels, out_channels=self.out_channels, stride=self.stride, dtype=self.dtype)
        self.activation_func = ACT2FN[self.config.hidden_act]

    def __call__(self, hidden_state: jnp.ndarray, deterministic: bool=True) -> jnp.ndarray:
        residual = hidden_state
        hidden_state = self.layer(hidden_state)
        residual = self.shortcut(residual, deterministic=deterministic)
        hidden_state += residual
        hidden_state = self.activation_func(hidden_state)
        return hidden_state
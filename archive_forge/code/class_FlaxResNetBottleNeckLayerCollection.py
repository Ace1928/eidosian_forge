from functools import partial
from typing import Optional, Tuple
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict
from ...modeling_flax_outputs import (
from ...modeling_flax_utils import (
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward
from .configuration_resnet import ResNetConfig
class FlaxResNetBottleNeckLayerCollection(nn.Module):
    out_channels: int
    stride: int = 1
    activation: Optional[str] = 'relu'
    reduction: int = 4
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        reduces_channels = self.out_channels // self.reduction
        self.layer = [FlaxResNetConvLayer(reduces_channels, kernel_size=1, dtype=self.dtype, name='0'), FlaxResNetConvLayer(reduces_channels, stride=self.stride, dtype=self.dtype, name='1'), FlaxResNetConvLayer(self.out_channels, kernel_size=1, activation=None, dtype=self.dtype, name='2')]

    def __call__(self, hidden_state: jnp.ndarray, deterministic: bool=True) -> jnp.ndarray:
        for layer in self.layer:
            hidden_state = layer(hidden_state, deterministic=deterministic)
        return hidden_state
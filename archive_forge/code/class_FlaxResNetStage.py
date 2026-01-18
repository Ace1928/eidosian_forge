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
class FlaxResNetStage(nn.Module):
    """
    A ResNet stage composed by stacked layers.
    """
    config: ResNetConfig
    in_channels: int
    out_channels: int
    stride: int = 2
    depth: int = 2
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.layers = FlaxResNetStageLayersCollection(self.config, in_channels=self.in_channels, out_channels=self.out_channels, stride=self.stride, depth=self.depth, dtype=self.dtype)

    def __call__(self, x: jnp.ndarray, deterministic: bool=True) -> jnp.ndarray:
        return self.layers(x, deterministic=deterministic)
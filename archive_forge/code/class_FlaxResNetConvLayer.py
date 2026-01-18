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
class FlaxResNetConvLayer(nn.Module):
    out_channels: int
    kernel_size: int = 3
    stride: int = 1
    activation: Optional[str] = 'relu'
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.convolution = nn.Conv(self.out_channels, kernel_size=(self.kernel_size, self.kernel_size), strides=self.stride, padding=self.kernel_size // 2, dtype=self.dtype, use_bias=False, kernel_init=nn.initializers.variance_scaling(2.0, mode='fan_out', distribution='normal', dtype=self.dtype))
        self.normalization = nn.BatchNorm(momentum=0.9, epsilon=1e-05, dtype=self.dtype)
        self.activation_func = ACT2FN[self.activation] if self.activation is not None else Identity()

    def __call__(self, x: jnp.ndarray, deterministic: bool=True) -> jnp.ndarray:
        hidden_state = self.convolution(x)
        hidden_state = self.normalization(hidden_state, use_running_average=deterministic)
        hidden_state = self.activation_func(hidden_state)
        return hidden_state
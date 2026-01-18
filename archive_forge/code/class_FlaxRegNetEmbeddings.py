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
class FlaxRegNetEmbeddings(nn.Module):
    config: RegNetConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.embedder = FlaxRegNetConvLayer(self.config.embedding_size, kernel_size=3, stride=2, activation=self.config.hidden_act, dtype=self.dtype)

    def __call__(self, pixel_values: jnp.ndarray, deterministic: bool=True) -> jnp.ndarray:
        num_channels = pixel_values.shape[-1]
        if num_channels != self.config.num_channels:
            raise ValueError('Make sure that the channel dimension of the pixel values match with the one set in the configuration.')
        hidden_state = self.embedder(pixel_values, deterministic=deterministic)
        return hidden_state
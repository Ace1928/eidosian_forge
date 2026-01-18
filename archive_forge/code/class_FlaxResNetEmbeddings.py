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
class FlaxResNetEmbeddings(nn.Module):
    """
    ResNet Embeddings (stem) composed of a single aggressive convolution.
    """
    config: ResNetConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.embedder = FlaxResNetConvLayer(self.config.embedding_size, kernel_size=7, stride=2, activation=self.config.hidden_act, dtype=self.dtype)
        self.max_pool = partial(nn.max_pool, window_shape=(3, 3), strides=(2, 2), padding=((1, 1), (1, 1)))

    def __call__(self, pixel_values: jnp.ndarray, deterministic: bool=True) -> jnp.ndarray:
        num_channels = pixel_values.shape[-1]
        if num_channels != self.config.num_channels:
            raise ValueError('Make sure that the channel dimension of the pixel values match with the one set in the configuration.')
        embedding = self.embedder(pixel_values, deterministic=deterministic)
        embedding = self.max_pool(embedding)
        return embedding
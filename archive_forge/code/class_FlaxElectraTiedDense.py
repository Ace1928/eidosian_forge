from typing import Callable, Optional, Tuple
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from flax.linen import partitioning as nn_partitioning
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from ...modeling_flax_outputs import (
from ...modeling_flax_utils import (
from ...utils import ModelOutput, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_electra import ElectraConfig
class FlaxElectraTiedDense(nn.Module):
    embedding_size: int
    dtype: jnp.dtype = jnp.float32
    precision = None
    bias_init: Callable[..., np.ndarray] = jax.nn.initializers.zeros

    def setup(self):
        self.bias = self.param('bias', self.bias_init, (self.embedding_size,))

    def __call__(self, x, kernel):
        x = jnp.asarray(x, self.dtype)
        kernel = jnp.asarray(kernel, self.dtype)
        y = lax.dot_general(x, kernel, (((x.ndim - 1,), (0,)), ((), ())), precision=self.precision)
        bias = jnp.asarray(self.bias, self.dtype)
        return y + bias
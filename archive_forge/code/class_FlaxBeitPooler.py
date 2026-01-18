from typing import Callable, List, Optional, Tuple
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from ...modeling_flax_outputs import (
from ...modeling_flax_utils import (
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward
from .configuration_beit import BeitConfig
class FlaxBeitPooler(nn.Module):
    config: BeitConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if self.config.use_mean_pooling:
            self.layernorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)

    def __call__(self, hidden_states):
        if self.config.use_mean_pooling:
            patch_tokens = hidden_states[:, 1:, :]
            pooled_output = self.layernorm(jnp.mean(patch_tokens, axis=1))
        else:
            pooled_output = hidden_states[:, 0]
        return pooled_output
from typing import Callable, Optional, Tuple
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from ...modeling_flax_outputs import (
from ...modeling_flax_utils import (
from ...utils import ModelOutput, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_albert import AlbertConfig
class FlaxAlbertOnlyMLMHead(nn.Module):
    config: AlbertConfig
    dtype: jnp.dtype = jnp.float32
    bias_init: Callable[..., np.ndarray] = jax.nn.initializers.zeros

    def setup(self):
        self.dense = nn.Dense(self.config.embedding_size, dtype=self.dtype)
        self.activation = ACT2FN[self.config.hidden_act]
        self.LayerNorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        self.decoder = nn.Dense(self.config.vocab_size, dtype=self.dtype, use_bias=False)
        self.bias = self.param('bias', self.bias_init, (self.config.vocab_size,))

    def __call__(self, hidden_states, shared_embedding=None):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        if shared_embedding is not None:
            hidden_states = self.decoder.apply({'params': {'kernel': shared_embedding.T}}, hidden_states)
        else:
            hidden_states = self.decoder(hidden_states)
        hidden_states += self.bias
        return hidden_states
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
class FlaxAlbertModule(nn.Module):
    config: AlbertConfig
    dtype: jnp.dtype = jnp.float32
    add_pooling_layer: bool = True

    def setup(self):
        self.embeddings = FlaxAlbertEmbeddings(self.config, dtype=self.dtype)
        self.encoder = FlaxAlbertEncoder(self.config, dtype=self.dtype)
        if self.add_pooling_layer:
            self.pooler = nn.Dense(self.config.hidden_size, kernel_init=jax.nn.initializers.normal(self.config.initializer_range), dtype=self.dtype, name='pooler')
            self.pooler_activation = nn.tanh
        else:
            self.pooler = None
            self.pooler_activation = None

    def __call__(self, input_ids, attention_mask, token_type_ids: Optional[np.ndarray]=None, position_ids: Optional[np.ndarray]=None, deterministic: bool=True, output_attentions: bool=False, output_hidden_states: bool=False, return_dict: bool=True):
        if token_type_ids is None:
            token_type_ids = jnp.zeros_like(input_ids)
        if position_ids is None:
            position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)
        hidden_states = self.embeddings(input_ids, token_type_ids, position_ids, deterministic=deterministic)
        outputs = self.encoder(hidden_states, attention_mask, deterministic=deterministic, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        hidden_states = outputs[0]
        if self.add_pooling_layer:
            pooled = self.pooler(hidden_states[:, 0])
            pooled = self.pooler_activation(pooled)
        else:
            pooled = None
        if not return_dict:
            if pooled is None:
                return (hidden_states,) + outputs[1:]
            return (hidden_states, pooled) + outputs[1:]
        return FlaxBaseModelOutputWithPooling(last_hidden_state=hidden_states, pooler_output=pooled, hidden_states=outputs.hidden_states, attentions=outputs.attentions)
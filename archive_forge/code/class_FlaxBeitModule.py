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
class FlaxBeitModule(nn.Module):
    config: BeitConfig
    dtype: jnp.dtype = jnp.float32
    add_pooling_layer: bool = True

    def setup(self):
        self.embeddings = FlaxBeitEmbeddings(self.config, dtype=self.dtype)
        self.encoder = FlaxBeitEncoder(self.config, window_size=self.embeddings.patch_embeddings.patch_shape, dtype=self.dtype)
        if not self.config.use_mean_pooling:
            self.layernorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        self.pooler = FlaxBeitPooler(self.config, dtype=self.dtype) if self.add_pooling_layer else None

    def __call__(self, pixel_values, bool_masked_pos=None, deterministic: bool=True, output_attentions: bool=False, output_hidden_states: bool=False, return_dict: bool=True):
        hidden_states = self.embeddings(pixel_values, bool_masked_pos, deterministic=deterministic)
        outputs = self.encoder(hidden_states, deterministic=deterministic, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        hidden_states = outputs[0]
        if not self.config.use_mean_pooling:
            hidden_states = self.layernorm(hidden_states)
        pooled = self.pooler(hidden_states) if self.add_pooling_layer else None
        if not return_dict:
            if pooled is None:
                return (hidden_states,) + outputs[1:]
            return (hidden_states, pooled) + outputs[1:]
        return FlaxBeitModelOutputWithPooling(last_hidden_state=hidden_states, pooler_output=pooled, hidden_states=outputs.hidden_states, attentions=outputs.attentions)
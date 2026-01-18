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
class FlaxAlbertLayerCollection(nn.Module):
    config: AlbertConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.layers = [FlaxAlbertLayer(self.config, name=str(i), dtype=self.dtype) for i in range(self.config.inner_group_num)]

    def __call__(self, hidden_states, attention_mask, deterministic: bool=True, output_attentions: bool=False, output_hidden_states: bool=False):
        layer_hidden_states = ()
        layer_attentions = ()
        for layer_index, albert_layer in enumerate(self.layers):
            layer_output = albert_layer(hidden_states, attention_mask, deterministic=deterministic, output_attentions=output_attentions)
            hidden_states = layer_output[0]
            if output_attentions:
                layer_attentions = layer_attentions + (layer_output[1],)
            if output_hidden_states:
                layer_hidden_states = layer_hidden_states + (hidden_states,)
        outputs = (hidden_states,)
        if output_hidden_states:
            outputs = outputs + (layer_hidden_states,)
        if output_attentions:
            outputs = outputs + (layer_attentions,)
        return outputs
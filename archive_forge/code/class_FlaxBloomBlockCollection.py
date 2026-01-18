import math
from functools import partial
from typing import Optional, Tuple
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, dot_product_attention_weights, make_causal_mask
from flax.linen.activation import tanh
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from ...modeling_flax_outputs import (
from ...modeling_flax_utils import FlaxPreTrainedModel, append_call_sample_docstring
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_bloom import BloomConfig
class FlaxBloomBlockCollection(nn.Module):
    config: BloomConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.layers = [FlaxBloomBlock(self.config, name=str(layer_number), dtype=self.dtype) for layer_number in range(self.config.num_hidden_layers)]

    def __call__(self, hidden_states, alibi, attention_mask=None, deterministic: bool=True, init_cache: bool=False, output_attentions: bool=False, output_hidden_states: bool=False):
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        for layer_number in range(self.config.num_hidden_layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            layer_outputs = self.layers[layer_number](hidden_states, alibi=alibi, attention_mask=attention_mask, deterministic=deterministic, init_cache=init_cache, output_attentions=output_attentions)
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions += (layer_outputs[1],)
        outputs = (hidden_states, all_hidden_states, all_attentions)
        return outputs
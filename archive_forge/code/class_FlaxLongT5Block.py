import copy
from typing import Any, Callable, List, Optional, Tuple
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from flax.linen import partitioning as nn_partitioning
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax.random import PRNGKey
from ...modeling_flax_outputs import (
from ...modeling_flax_utils import (
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_longt5 import LongT5Config
class FlaxLongT5Block(nn.Module):
    config: LongT5Config
    has_relative_attention_bias: bool = False
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.causal = self.config.causal
        if self.causal:
            attention_layer = FlaxLongT5LayerSelfAttention
        elif self.config.encoder_attention_type == 'local':
            attention_layer = FlaxLongT5LayerLocalSelfAttention
        elif self.config.encoder_attention_type == 'transient-global':
            attention_layer = FlaxLongT5LayerTransientGlobalSelfAttention
        else:
            raise ValueError(f'For encoder attention mechanism, either `local` or `transient-global` attention type is expected, but got {self.config.encoder_attention_type}.')
        self.layer = (attention_layer(self.config, has_relative_attention_bias=self.has_relative_attention_bias, name=str(0), dtype=self.dtype),)
        feed_forward_index = 1
        if self.causal:
            self.layer += (FlaxLongT5LayerCrossAttention(self.config, name=str(1), dtype=self.dtype),)
            feed_forward_index += 1
        self.layer += (FlaxLongT5LayerFF(self.config, name=str(feed_forward_index), dtype=self.dtype),)

    def __call__(self, hidden_states, attention_mask=None, position_bias=None, encoder_hidden_states=None, encoder_attention_mask=None, encoder_decoder_position_bias=None, output_attentions=False, return_dict=True, deterministic=True, init_cache=False):
        self_attention_outputs = self.layer[0](hidden_states, attention_mask=attention_mask, position_bias=position_bias, output_attentions=output_attentions, deterministic=deterministic, init_cache=init_cache)
        hidden_states = self_attention_outputs[0]
        attention_outputs = self_attention_outputs[1:]
        do_cross_attention = self.causal and encoder_hidden_states is not None
        if do_cross_attention:
            cross_attention_outputs = self.layer[1](hidden_states, key_value_states=encoder_hidden_states, attention_mask=encoder_attention_mask, position_bias=encoder_decoder_position_bias, output_attentions=output_attentions, deterministic=deterministic)
            hidden_states = cross_attention_outputs[0]
            attention_outputs = attention_outputs + cross_attention_outputs[1:]
        hidden_states = self.layer[-1](hidden_states, deterministic=deterministic)
        outputs = (hidden_states,)
        outputs = outputs + attention_outputs
        return outputs
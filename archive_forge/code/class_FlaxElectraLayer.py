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
class FlaxElectraLayer(nn.Module):
    config: ElectraConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.attention = FlaxElectraAttention(self.config, causal=self.config.is_decoder, dtype=self.dtype)
        self.intermediate = FlaxElectraIntermediate(self.config, dtype=self.dtype)
        self.output = FlaxElectraOutput(self.config, dtype=self.dtype)
        if self.config.add_cross_attention:
            self.crossattention = FlaxElectraAttention(self.config, causal=False, dtype=self.dtype)

    def __call__(self, hidden_states, attention_mask, layer_head_mask, encoder_hidden_states: Optional[jnp.ndarray]=None, encoder_attention_mask: Optional[jnp.ndarray]=None, init_cache: bool=False, deterministic: bool=True, output_attentions: bool=False):
        attention_outputs = self.attention(hidden_states, attention_mask, layer_head_mask=layer_head_mask, init_cache=init_cache, deterministic=deterministic, output_attentions=output_attentions)
        attention_output = attention_outputs[0]
        if encoder_hidden_states is not None:
            cross_attention_outputs = self.crossattention(attention_output, attention_mask=encoder_attention_mask, layer_head_mask=layer_head_mask, key_value_states=encoder_hidden_states, deterministic=deterministic, output_attentions=output_attentions)
            attention_output = cross_attention_outputs[0]
        hidden_states = self.intermediate(attention_output)
        hidden_states = self.output(hidden_states, attention_output, deterministic=deterministic)
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attention_outputs[1],)
            if encoder_hidden_states is not None:
                outputs += (cross_attention_outputs[1],)
        return outputs
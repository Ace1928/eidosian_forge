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
class FlaxLongT5Stack(nn.Module):
    config: LongT5Config
    embed_tokens: nn.Embed
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    def setup(self):
        self.causal = self.config.causal
        self.block = FlaxLongT5BlockCollection(self.config, dtype=self.dtype, gradient_checkpointing=self.gradient_checkpointing)
        self.final_layer_norm = FlaxLongT5LayerNorm(self.config.d_model, eps=self.config.layer_norm_epsilon, dtype=self.dtype)
        self.dropout = nn.Dropout(self.config.dropout_rate)

    def __call__(self, input_ids=None, attention_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, output_attentions: bool=False, output_hidden_states: bool=False, return_dict: bool=True, deterministic: bool=True, init_cache: bool=False):
        hidden_states = self.embed_tokens(input_ids)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        outputs = self.block(hidden_states, attention_mask=attention_mask, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, deterministic=deterministic, init_cache=init_cache)
        hidden_states = outputs[0]
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        all_hidden_states = None
        if output_hidden_states:
            all_hidden_states = outputs.hidden_states
            all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            if output_hidden_states:
                return (hidden_states, all_hidden_states) + outputs[2:]
            return (hidden_states,) + outputs[1:]
        return FlaxBaseModelOutputWithPastAndCrossAttentions(last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=outputs.attentions, cross_attentions=outputs.cross_attentions)
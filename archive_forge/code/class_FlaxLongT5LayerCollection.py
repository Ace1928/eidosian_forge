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
class FlaxLongT5LayerCollection(nn.Module):
    config: LongT5Config
    has_relative_attention_bias: bool
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.layer = FlaxLongT5Block(self.config, has_relative_attention_bias=self.has_relative_attention_bias, dtype=self.dtype)

    def __call__(self, hidden_states, attention_mask=None, position_bias=None, encoder_hidden_states=None, encoder_attention_mask=None, encoder_decoder_position_bias=None, output_attentions=False, deterministic=True, init_cache=False):
        return self.layer(hidden_states, attention_mask=attention_mask, position_bias=position_bias, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask, encoder_decoder_position_bias=encoder_decoder_position_bias, output_attentions=output_attentions, deterministic=deterministic, init_cache=init_cache)
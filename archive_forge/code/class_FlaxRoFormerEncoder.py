from typing import Callable, Optional, Tuple
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from ...modeling_flax_outputs import (
from ...modeling_flax_utils import ACT2FN, FlaxPreTrainedModel, append_call_sample_docstring, overwrite_call_docstring
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_roformer import RoFormerConfig
class FlaxRoFormerEncoder(nn.Module):
    config: RoFormerConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.embed_positions = create_sinusoidal_positions(self.config.max_position_embeddings, self.config.hidden_size // self.config.num_attention_heads)
        self.layer = FlaxRoFormerLayerCollection(self.config, dtype=self.dtype)

    def __call__(self, hidden_states, attention_mask, head_mask, deterministic: bool=True, output_attentions: bool=False, output_hidden_states: bool=False, return_dict: bool=True):
        sinusoidal_pos = self.embed_positions[:hidden_states.shape[1], :]
        return self.layer(hidden_states, attention_mask, sinusoidal_pos, head_mask, deterministic=deterministic, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
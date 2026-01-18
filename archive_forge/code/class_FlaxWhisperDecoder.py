import math
import random
from functools import partial
from typing import Optional, Tuple
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from flax.linen import partitioning as nn_partitioning
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from jax.random import PRNGKey
from ...generation.flax_logits_process import FlaxWhisperTimeStampLogitsProcessor
from ...modeling_flax_outputs import (
from ...modeling_flax_utils import (
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_whisper import WhisperConfig
class FlaxWhisperDecoder(nn.Module):
    config: WhisperConfig
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    def setup(self) -> None:
        self.embed_tokens = nn.Embed(self.config.vocab_size, self.config.d_model, dtype=self.dtype)
        self.embed_positions = nn.Embed(self.config.max_target_positions, self.config.d_model, dtype=self.dtype)
        self.layers = FlaxWhisperDecoderLayerCollection(self.config, dtype=self.dtype, gradient_checkpointing=self.gradient_checkpointing)
        self.dropout_layer = nn.Dropout(rate=self.config.dropout)
        self.layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=1e-05)

    def __call__(self, input_ids: jnp.ndarray, attention_mask: jnp.ndarray, position_ids: jnp.ndarray, encoder_hidden_states: Optional[jnp.ndarray]=None, init_cache: bool=False, output_attentions: bool=False, output_hidden_states: bool=False, return_dict: bool=True, deterministic: bool=True) -> Tuple[jnp.ndarray]:
        input_embeds = self.embed_tokens(input_ids)
        position_embeds = self.embed_positions(position_ids)
        hidden_states = input_embeds + position_embeds
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)
        outputs = self.layers(hidden_states, attention_mask=attention_mask, encoder_hidden_states=encoder_hidden_states, deterministic=deterministic, init_cache=init_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        last_hidden_states = outputs[0]
        last_hidden_states = self.layer_norm(last_hidden_states)
        hidden_states = None
        if output_hidden_states:
            hidden_states = outputs[1]
            hidden_states = hidden_states[:-1] + (last_hidden_states,)
        if not return_dict:
            outputs = (last_hidden_states, hidden_states) + (outputs[2:] if output_hidden_states else outputs[1:])
            return tuple((v for v in outputs if v is not None))
        return FlaxBaseModelOutputWithPastAndCrossAttentions(last_hidden_state=last_hidden_states, hidden_states=hidden_states, attentions=outputs.attentions, cross_attentions=outputs.cross_attentions)
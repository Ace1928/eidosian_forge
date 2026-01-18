import math
from typing import Callable, Optional, Tuple
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from ...modeling_flax_outputs import (
from ...modeling_flax_utils import ACT2FN, FlaxPreTrainedModel, append_call_sample_docstring, overwrite_call_docstring
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_distilbert import DistilBertConfig
class FlaxEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""
    config: DistilBertConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.word_embeddings = nn.Embed(self.config.vocab_size, self.config.dim, embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range))
        if not self.config.sinusoidal_pos_embds:
            self.position_embeddings = nn.Embed(self.config.max_position_embeddings, self.config.dim, embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range))
        else:
            self.pos_encoding = positional_encoding(self.config.max_position_embeddings, self.config.dim)
        self.LayerNorm = nn.LayerNorm(epsilon=1e-12, dtype=self.dtype)
        self.dropout = nn.Dropout(rate=self.config.dropout)

    def __call__(self, input_ids, deterministic: bool=True):
        batch_size, seq_length = input_ids.shape
        inputs_embeds = self.word_embeddings(input_ids.astype('i4'))
        if not self.config.sinusoidal_pos_embds:
            position_ids = jnp.arange(seq_length).astype('i4')
            position_ids = jnp.broadcast_to(position_ids, shape=(batch_size, seq_length))
            position_embeds = self.position_embeddings(position_ids.astype('i4'))
        else:
            position_embeds = self.pos_encoding[:, :seq_length, :]
            position_embeds = position_embeds.astype(inputs_embeds.dtype)
        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        return hidden_states
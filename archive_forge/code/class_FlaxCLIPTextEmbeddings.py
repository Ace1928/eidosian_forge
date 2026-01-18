from typing import Any, Optional, Tuple, Union
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from ...modeling_flax_outputs import FlaxBaseModelOutput, FlaxBaseModelOutputWithPooling
from ...modeling_flax_utils import (
from ...utils import ModelOutput, add_start_docstrings, logging
from .configuration_clip import CLIPConfig, CLIPTextConfig, CLIPVisionConfig
class FlaxCLIPTextEmbeddings(nn.Module):
    config: CLIPTextConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        embed_dim = self.config.hidden_size
        self.token_embedding = nn.Embed(self.config.vocab_size, embed_dim, embedding_init=jax.nn.initializers.normal())
        self.position_embedding = nn.Embed(self.config.max_position_embeddings, embed_dim, embedding_init=jax.nn.initializers.normal())
        self.position_ids = jnp.expand_dims(jnp.arange(0, self.config.max_position_embeddings, dtype='i4'), axis=(0, 1))

    def __call__(self, input_ids, position_ids):
        input_embeds = self.token_embedding(input_ids.astype('i4'))
        position_embeds = self.position_embedding(position_ids.astype('i4'))
        embeddings = input_embeds + position_embeds
        return embeddings
import math
import random
from functools import partial
from typing import Callable, Optional, Tuple
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from jax.random import PRNGKey
from ...modeling_flax_outputs import (
from ...modeling_flax_utils import (
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_bart import BartConfig
class FlaxBartDecoderWrapper(nn.Module):
    """
    This wrapper class is a helper class to correctly load pretrained checkpoints when the causal language model is
    used in combination with the [`EncoderDecoderModel`] framework.
    """
    config: BartConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        embed_dim = self.config.d_model
        embed_tokens = nn.Embed(self.config.vocab_size, embed_dim, embedding_init=jax.nn.initializers.normal(self.config.init_std), dtype=self.dtype)
        self.decoder = FlaxBartDecoder(config=self.config, embed_tokens=embed_tokens, dtype=self.dtype)

    def __call__(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)
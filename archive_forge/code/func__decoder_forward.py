import math
import random
from functools import partial
from typing import Callable, Optional, Tuple
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from jax.random import PRNGKey
from ...modeling_flax_outputs import (
from ...modeling_flax_utils import (
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_marian import MarianConfig
def _decoder_forward(module, decoder_input_ids, decoder_attention_mask, decoder_position_ids, **kwargs):
    decoder_module = module._get_decoder_module()
    outputs = decoder_module(decoder_input_ids, decoder_attention_mask, decoder_position_ids, **kwargs)
    hidden_states = outputs[0]
    if self.config.tie_word_embeddings:
        shared_embedding = module.model.variables['params']['shared']['embedding']
        lm_logits = module.lm_head.apply({'params': {'kernel': shared_embedding.T}}, hidden_states)
    else:
        lm_logits = module.lm_head(hidden_states)
    lm_logits += module.final_logits_bias.astype(self.dtype)
    return (lm_logits, outputs)
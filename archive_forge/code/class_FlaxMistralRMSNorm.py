from typing import Optional, Tuple
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from ...modeling_flax_outputs import (
from ...modeling_flax_utils import ACT2FN, FlaxPreTrainedModel, append_call_sample_docstring, logging
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward
from .configuration_mistral import MistralConfig
class FlaxMistralRMSNorm(nn.Module):
    config: MistralConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.epsilon = self.config.rms_norm_eps
        self.weight = self.param('weight', lambda _, shape: jnp.ones(shape), self.config.hidden_size)

    def __call__(self, hidden_states):
        variance = jnp.asarray(hidden_states, dtype=jnp.float32)
        variance = jnp.power(variance, 2)
        variance = variance.mean(-1, keepdims=True)
        hidden_states = hidden_states / jnp.sqrt(variance + self.epsilon)
        return self.weight * jnp.asarray(hidden_states, dtype=self.dtype)
from functools import partial
from typing import Optional, Tuple, Union
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from ...modeling_flax_outputs import FlaxBaseModelOutput, FlaxCausalLMOutput
from ...modeling_flax_utils import (
from ...utils import ModelOutput, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_wav2vec2 import Wav2Vec2Config
class FlaxWav2Vec2GumbelVectorQuantizer(nn.Module):
    """
    Vector quantization using gumbel softmax. See [CATEGORICAL REPARAMETERIZATION WITH
    GUMBEL-SOFTMAX](https://arxiv.org/pdf/1611.01144.pdf) for more information.
    """
    config: Wav2Vec2Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.num_groups = self.config.num_codevector_groups
        self.num_vars = self.config.num_codevectors_per_group
        if self.config.codevector_dim % self.num_groups != 0:
            raise ValueError(f'`config.codevector_dim {self.config.codevector_dim} must be divisible by `config.num_codevector_groups` {self.num_groups} for concatenation')
        self.codevectors = self.param('codevectors', jax.nn.initializers.uniform(), (1, self.num_groups * self.num_vars, self.config.codevector_dim // self.num_groups))
        self.weight_proj = nn.Dense(self.num_groups * self.num_vars, kernel_init=jax.nn.initializers.normal(1.0), dtype=self.dtype)

    @staticmethod
    def _compute_perplexity(probs, mask=None):
        if mask is not None:
            mask_extended = jnp.broadcast_to(mask.flatten()[:, None, None], probs.shape)
            probs = jnp.where(mask_extended, probs, jnp.zeros_like(probs))
            marginal_probs = probs.sum(axis=0) / mask.sum()
        else:
            marginal_probs = probs.mean(axis=0)
        perplexity = jnp.exp(-jnp.sum(marginal_probs * jnp.log(marginal_probs + 1e-07), axis=-1)).sum()
        return perplexity

    def __call__(self, hidden_states, mask_time_indices=None, deterministic=True, temperature=1):
        batch_size, sequence_length, hidden_size = hidden_states.shape
        hidden_states = self.weight_proj(hidden_states)
        hidden_states = hidden_states.reshape(batch_size * sequence_length * self.num_groups, -1)
        if not deterministic:
            gumbel_rng = self.make_rng('gumbel')
            gumbels = jax.random.gumbel(gumbel_rng, hidden_states.shape)
            codevector_probs = nn.softmax((hidden_states + gumbels) / temperature)
            codevector_soft_dist = nn.softmax(hidden_states.reshape(batch_size * sequence_length, self.num_groups, -1), axis=-1)
            perplexity = self._compute_perplexity(codevector_soft_dist, mask_time_indices)
        else:
            codevector_idx = hidden_states.argmax(axis=-1)
            codevector_probs = jax.nn.one_hot(codevector_idx, hidden_states.shape[-1]) * 1.0
            codevector_probs = codevector_probs.reshape(batch_size * sequence_length, self.num_groups, -1)
            perplexity = self._compute_perplexity(codevector_probs, mask_time_indices)
        codevector_probs = codevector_probs.reshape(batch_size * sequence_length, -1)
        codevectors_per_group = jnp.expand_dims(codevector_probs, axis=-1) * self.codevectors
        codevectors = codevectors_per_group.reshape(batch_size * sequence_length, self.num_groups, self.num_vars, -1)
        codevectors = codevectors.sum(-2).reshape(batch_size, sequence_length, -1)
        return (codevectors, perplexity)
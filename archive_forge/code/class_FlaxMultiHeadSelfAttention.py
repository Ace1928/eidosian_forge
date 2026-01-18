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
class FlaxMultiHeadSelfAttention(nn.Module):
    config: DistilBertConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.n_heads = self.config.n_heads
        self.dim = self.config.dim
        self.dropout = nn.Dropout(rate=self.config.attention_dropout)
        if not self.dim % self.n_heads == 0:
            raise ValueError(f'Hidden size {self.dim} not dividable by number of heads {self.n_heads}')
        self.q_lin = nn.Dense(self.dim, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range))
        self.k_lin = nn.Dense(self.dim, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range))
        self.v_lin = nn.Dense(self.dim, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range))
        self.out_lin = nn.Dense(self.dim, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range))

    def __call__(self, query, key, value, mask, deterministic: bool=True, output_attentions: bool=False):
        bs, q_len, dim = query.shape
        k_len = key.shape[1]
        dim_per_head = self.dim // self.n_heads
        mask_reshp = (bs, 1, 1, k_len)

        def shape(x):
            """separate heads"""
            return x.reshape(bs, -1, self.n_heads, dim_per_head).transpose(0, 2, 1, 3)

        def unshape(x):
            """group heads"""
            return x.transpose(0, 2, 1, 3).reshape(bs, -1, self.n_heads * dim_per_head)
        q = shape(self.q_lin(query))
        k = shape(self.k_lin(key))
        v = shape(self.v_lin(value))
        q = q / math.sqrt(dim_per_head)
        scores = jnp.matmul(q, k.transpose(0, 1, 3, 2))
        mask = jnp.reshape(mask, mask_reshp)
        mask = mask.astype(scores.dtype)
        scores = scores - 1e+30 * (1.0 - mask)
        weights = nn.softmax(scores, axis=-1)
        weights = self.dropout(weights, deterministic=deterministic)
        context = jnp.matmul(weights, v)
        context = unshape(context)
        context = self.out_lin(context)
        if output_attentions:
            return (context, weights)
        else:
            return (context,)
from typing import Callable, Optional, Tuple
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from flax.linen import partitioning as nn_partitioning
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from ...modeling_flax_outputs import (
from ...modeling_flax_utils import ACT2FN, FlaxPreTrainedModel, append_call_sample_docstring, overwrite_call_docstring
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_roberta import RobertaConfig
class FlaxRobertaClassificationHead(nn.Module):
    config: RobertaConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.dense = nn.Dense(self.config.hidden_size, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(self.config.initializer_range))
        classifier_dropout = self.config.classifier_dropout if self.config.classifier_dropout is not None else self.config.hidden_dropout_prob
        self.dropout = nn.Dropout(rate=classifier_dropout)
        self.out_proj = nn.Dense(self.config.num_labels, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(self.config.initializer_range))

    def __call__(self, hidden_states, deterministic=True):
        hidden_states = hidden_states[:, 0, :]
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        hidden_states = self.dense(hidden_states)
        hidden_states = nn.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states
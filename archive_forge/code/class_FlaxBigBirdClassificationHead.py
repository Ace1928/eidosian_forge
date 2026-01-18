from typing import Callable, Optional, Tuple
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from flax.linen import partitioning as nn_partitioning
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from ...modeling_flax_outputs import (
from ...modeling_flax_utils import (
from ...utils import ModelOutput, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_big_bird import BigBirdConfig
class FlaxBigBirdClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""
    config: BigBirdConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.dense = nn.Dense(self.config.hidden_size, dtype=self.dtype)
        classifier_dropout = self.config.classifier_dropout if self.config.classifier_dropout is not None else self.config.hidden_dropout_prob
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Dense(self.config.num_labels, dtype=self.dtype)

    def __call__(self, features, deterministic=True):
        x = features[:, 0, :]
        x = self.dropout(x, deterministic=deterministic)
        x = self.dense(x)
        x = ACT2FN[self.config.hidden_act](x)
        x = self.dropout(x, deterministic=deterministic)
        x = self.out_proj(x)
        return x
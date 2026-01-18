from typing import Callable, Optional, Tuple
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from ...modeling_flax_outputs import (
from ...modeling_flax_utils import (
from ...utils import ModelOutput, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_albert import AlbertConfig
class FlaxAlbertSOPHead(nn.Module):
    config: AlbertConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.dropout = nn.Dropout(self.config.classifier_dropout_prob)
        self.classifier = nn.Dense(2, dtype=self.dtype)

    def __call__(self, pooled_output, deterministic=True):
        pooled_output = self.dropout(pooled_output, deterministic=deterministic)
        logits = self.classifier(pooled_output)
        return logits
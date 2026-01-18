from typing import Callable, Optional, Tuple
import flax
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
from ...modeling_flax_utils import (
from ...utils import ModelOutput, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_bert import BertConfig
class FlaxBertOnlyNSPHead(nn.Module):
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.seq_relationship = nn.Dense(2, dtype=self.dtype)

    def __call__(self, pooled_output):
        return self.seq_relationship(pooled_output)
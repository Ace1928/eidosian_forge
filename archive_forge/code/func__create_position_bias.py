import copy
from typing import Any, Callable, List, Optional, Tuple
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from flax.linen import partitioning as nn_partitioning
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax.random import PRNGKey
from ...modeling_flax_outputs import (
from ...modeling_flax_utils import (
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_longt5 import LongT5Config
def _create_position_bias(self, block_len: int, attention_mask: Optional[np.ndarray]) -> np.ndarray:
    if self.has_relative_attention_bias:
        position_bias = self.compute_bias(block_len)
    elif attention_mask is not None:
        position_bias = jnp.zeros_like(attention_mask)
    else:
        position_bias = jnp.zeros((1, 1, self.n_heads, block_len, 3 * block_len), dtype=self.dtype)
    return position_bias
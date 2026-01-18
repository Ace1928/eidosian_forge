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
from .configuration_mbart import MBartConfig
@add_start_docstrings('The bare MBart Model transformer outputting raw hidden-states without any specific head on top.', MBART_START_DOCSTRING)
class FlaxMBartModel(FlaxMBartPreTrainedModel):
    config: MBartConfig
    dtype: jnp.dtype = jnp.float32
    module_class = FlaxMBartModule
import math
import random
from functools import partial
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
from jax.random import PRNGKey
from ...modeling_flax_outputs import (
from ...modeling_flax_utils import ACT2FN, FlaxPreTrainedModel, append_call_sample_docstring
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_xglm import XGLMConfig
@add_start_docstrings('The bare XGLM Model transformer outputting raw hidden-states without any specific head on top.', XGLM_START_DOCSTRING)
class FlaxXGLMModel(FlaxXGLMPreTrainedModel):
    module_class = FlaxXGLMModule
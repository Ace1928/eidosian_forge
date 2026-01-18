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
def _get_normed_weights(self):
    weight_v_norm = jnp.linalg.norm(self.weight_v, axis=(0, 1))[None, None, :]
    normed_weight_v = jnp.divide(self.weight_v, weight_v_norm)
    normed_kernel = jnp.multiply(normed_weight_v, self.weight_g)
    return normed_kernel
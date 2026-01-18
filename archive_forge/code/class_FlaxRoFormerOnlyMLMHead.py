from typing import Callable, Optional, Tuple
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from ...modeling_flax_outputs import (
from ...modeling_flax_utils import ACT2FN, FlaxPreTrainedModel, append_call_sample_docstring, overwrite_call_docstring
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_roformer import RoFormerConfig
class FlaxRoFormerOnlyMLMHead(nn.Module):
    config: RoFormerConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.predictions = FlaxRoFormerLMPredictionHead(self.config, dtype=self.dtype)

    def __call__(self, hidden_states, shared_embedding=None):
        hidden_states = self.predictions(hidden_states, shared_embedding=shared_embedding)
        return hidden_states
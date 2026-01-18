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
class FlaxBigBirdForQuestionAnsweringHead(nn.Module):
    config: BigBirdConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)
        self.intermediate = FlaxBigBirdIntermediate(self.config, dtype=self.dtype)
        self.output = FlaxBigBirdOutput(self.config, dtype=self.dtype)
        self.qa_outputs = nn.Dense(self.config.num_labels, dtype=self.dtype)

    def __call__(self, encoder_output, deterministic=True):
        hidden_states = self.dropout(encoder_output, deterministic=deterministic)
        hidden_states = self.intermediate(hidden_states)
        hidden_states = self.output(hidden_states, encoder_output)
        hidden_states = self.qa_outputs(hidden_states)
        return hidden_states
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
class FlaxBigBirdPreTrainingHeads(nn.Module):
    config: BigBirdConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.predictions = FlaxBigBirdLMPredictionHead(self.config, dtype=self.dtype)
        self.seq_relationship = nn.Dense(2, dtype=self.dtype)

    def __call__(self, hidden_states, pooled_output, shared_embedding=None):
        prediction_scores = self.predictions(hidden_states, shared_embedding=shared_embedding)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return (prediction_scores, seq_relationship_score)
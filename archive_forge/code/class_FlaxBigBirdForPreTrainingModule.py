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
class FlaxBigBirdForPreTrainingModule(nn.Module):
    config: BigBirdConfig
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    def setup(self):
        self.bert = FlaxBigBirdModule(config=self.config, dtype=self.dtype, gradient_checkpointing=self.gradient_checkpointing)
        self.cls = FlaxBigBirdPreTrainingHeads(config=self.config, dtype=self.dtype)

    def __call__(self, input_ids, attention_mask, token_type_ids, position_ids, head_mask, deterministic: bool=True, output_attentions: bool=False, output_hidden_states: bool=False, return_dict: bool=True):
        outputs = self.bert(input_ids, attention_mask, token_type_ids, position_ids, head_mask, deterministic=deterministic, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        if self.config.tie_word_embeddings:
            shared_embedding = self.bert.variables['params']['embeddings']['word_embeddings']['embedding']
        else:
            shared_embedding = None
        hidden_states = outputs[0]
        pooled_output = outputs[1]
        prediction_scores, seq_relationship_score = self.cls(hidden_states, pooled_output, shared_embedding=shared_embedding)
        if not return_dict:
            return (prediction_scores, seq_relationship_score) + outputs[2:]
        return FlaxBigBirdForPreTrainingOutput(prediction_logits=prediction_scores, seq_relationship_logits=seq_relationship_score, hidden_states=outputs.hidden_states, attentions=outputs.attentions)
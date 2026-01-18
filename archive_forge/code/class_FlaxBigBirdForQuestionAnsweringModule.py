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
class FlaxBigBirdForQuestionAnsweringModule(nn.Module):
    config: BigBirdConfig
    dtype: jnp.dtype = jnp.float32
    add_pooling_layer: bool = False
    gradient_checkpointing: bool = False

    def setup(self):
        self.config.num_labels = 2
        self.bert = FlaxBigBirdModule(self.config, dtype=self.dtype, add_pooling_layer=self.add_pooling_layer, gradient_checkpointing=self.gradient_checkpointing)
        self.qa_classifier = FlaxBigBirdForQuestionAnsweringHead(self.config, dtype=self.dtype)

    def __call__(self, input_ids, attention_mask, token_type_ids, position_ids, head_mask, logits_mask=None, deterministic: bool=True, output_attentions: bool=False, output_hidden_states: bool=False, return_dict: bool=True):
        outputs = self.bert(input_ids, attention_mask, token_type_ids, position_ids, head_mask, deterministic=deterministic, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        hidden_states = outputs[0]
        pooled_output = outputs[1] if self.add_pooling_layer else None
        logits = self.qa_classifier(hidden_states, deterministic=deterministic)
        if logits_mask is not None:
            logits = logits - logits_mask * 1000000.0
        start_logits, end_logits = logits.split(self.config.num_labels, axis=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        if not return_dict:
            return (start_logits, end_logits) + outputs[1:]
        return FlaxBigBirdForQuestionAnsweringModelOutput(start_logits=start_logits, end_logits=end_logits, pooled_output=pooled_output, hidden_states=outputs.hidden_states, attentions=outputs.attentions)
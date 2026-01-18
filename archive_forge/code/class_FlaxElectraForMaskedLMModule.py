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
from .configuration_electra import ElectraConfig
class FlaxElectraForMaskedLMModule(nn.Module):
    config: ElectraConfig
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    def setup(self):
        self.electra = FlaxElectraModule(config=self.config, dtype=self.dtype, gradient_checkpointing=self.gradient_checkpointing)
        self.generator_predictions = FlaxElectraGeneratorPredictions(config=self.config, dtype=self.dtype)
        if self.config.tie_word_embeddings:
            self.generator_lm_head = FlaxElectraTiedDense(self.config.vocab_size, dtype=self.dtype)
        else:
            self.generator_lm_head = nn.Dense(self.config.vocab_size, dtype=self.dtype)

    def __call__(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, deterministic: bool=True, output_attentions: bool=False, output_hidden_states: bool=False, return_dict: bool=True):
        outputs = self.electra(input_ids, attention_mask, token_type_ids, position_ids, head_mask, deterministic=deterministic, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        hidden_states = outputs[0]
        prediction_scores = self.generator_predictions(hidden_states)
        if self.config.tie_word_embeddings:
            shared_embedding = self.electra.variables['params']['embeddings']['word_embeddings']['embedding']
            prediction_scores = self.generator_lm_head(prediction_scores, shared_embedding.T)
        else:
            prediction_scores = self.generator_lm_head(prediction_scores)
        if not return_dict:
            return (prediction_scores,) + outputs[1:]
        return FlaxMaskedLMOutput(logits=prediction_scores, hidden_states=outputs.hidden_states, attentions=outputs.attentions)
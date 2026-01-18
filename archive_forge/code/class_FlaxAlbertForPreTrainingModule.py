from typing import Callable, Optional, Tuple
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from ...modeling_flax_outputs import (
from ...modeling_flax_utils import (
from ...utils import ModelOutput, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_albert import AlbertConfig
class FlaxAlbertForPreTrainingModule(nn.Module):
    config: AlbertConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.albert = FlaxAlbertModule(config=self.config, dtype=self.dtype)
        self.predictions = FlaxAlbertOnlyMLMHead(config=self.config, dtype=self.dtype)
        self.sop_classifier = FlaxAlbertSOPHead(config=self.config, dtype=self.dtype)

    def __call__(self, input_ids, attention_mask, token_type_ids, position_ids, deterministic: bool=True, output_attentions: bool=False, output_hidden_states: bool=False, return_dict: bool=True):
        outputs = self.albert(input_ids, attention_mask, token_type_ids, position_ids, deterministic=deterministic, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        if self.config.tie_word_embeddings:
            shared_embedding = self.albert.variables['params']['embeddings']['word_embeddings']['embedding']
        else:
            shared_embedding = None
        hidden_states = outputs[0]
        pooled_output = outputs[1]
        prediction_scores = self.predictions(hidden_states, shared_embedding=shared_embedding)
        sop_scores = self.sop_classifier(pooled_output, deterministic=deterministic)
        if not return_dict:
            return (prediction_scores, sop_scores) + outputs[2:]
        return FlaxAlbertForPreTrainingOutput(prediction_logits=prediction_scores, sop_logits=sop_scores, hidden_states=outputs.hidden_states, attentions=outputs.attentions)
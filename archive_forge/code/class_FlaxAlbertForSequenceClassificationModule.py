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
class FlaxAlbertForSequenceClassificationModule(nn.Module):
    config: AlbertConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.albert = FlaxAlbertModule(config=self.config, dtype=self.dtype)
        classifier_dropout = self.config.classifier_dropout_prob if self.config.classifier_dropout_prob is not None else self.config.hidden_dropout_prob
        self.dropout = nn.Dropout(rate=classifier_dropout)
        self.classifier = nn.Dense(self.config.num_labels, dtype=self.dtype)

    def __call__(self, input_ids, attention_mask, token_type_ids, position_ids, deterministic: bool=True, output_attentions: bool=False, output_hidden_states: bool=False, return_dict: bool=True):
        outputs = self.albert(input_ids, attention_mask, token_type_ids, position_ids, deterministic=deterministic, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output, deterministic=deterministic)
        logits = self.classifier(pooled_output)
        if not return_dict:
            return (logits,) + outputs[2:]
        return FlaxSequenceClassifierOutput(logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)
import math
from typing import Callable, Optional, Tuple
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from ...modeling_flax_outputs import (
from ...modeling_flax_utils import ACT2FN, FlaxPreTrainedModel, append_call_sample_docstring, overwrite_call_docstring
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_distilbert import DistilBertConfig
class FlaxDistilBertForMaskedLMModule(nn.Module):
    config: DistilBertConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.distilbert = FlaxDistilBertModule(self.config, dtype=self.dtype)
        self.vocab_transform = nn.Dense(self.config.dim, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range))
        self.vocab_layer_norm = nn.LayerNorm(epsilon=1e-12, dtype=self.dtype)
        if self.config.tie_word_embeddings:
            self.vocab_projector = FlaxDistilBertLMDecoder(self.config, dtype=self.dtype)
        else:
            self.vocab_projector = nn.Dense(self.config.vocab_size, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range))

    def __call__(self, input_ids, attention_mask, deterministic: bool=True, output_attentions: bool=False, output_hidden_states: bool=False, return_dict: bool=True):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        dlbrt_output = self.distilbert(input_ids=input_ids, attention_mask=attention_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, deterministic=deterministic, return_dict=return_dict)
        hidden_states = dlbrt_output[0]
        prediction_logits = self.vocab_transform(hidden_states)
        prediction_logits = ACT2FN[self.config.activation](prediction_logits)
        prediction_logits = self.vocab_layer_norm(prediction_logits)
        if self.config.tie_word_embeddings:
            shared_embedding = self.distilbert.variables['params']['embeddings']['word_embeddings']['embedding']
            prediction_logits = self.vocab_projector(prediction_logits, shared_embedding.T)
        else:
            prediction_logits = self.vocab_projector(prediction_logits)
        if not return_dict:
            output = (prediction_logits,) + dlbrt_output[1:]
            return output
        return FlaxMaskedLMOutput(logits=prediction_logits, hidden_states=dlbrt_output.hidden_states, attentions=dlbrt_output.attentions)
import math
import random
from functools import partial
from typing import Optional, Tuple
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from flax.linen import partitioning as nn_partitioning
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from jax.random import PRNGKey
from ...generation.flax_logits_process import FlaxWhisperTimeStampLogitsProcessor
from ...modeling_flax_outputs import (
from ...modeling_flax_utils import (
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_whisper import WhisperConfig
class FlaxWhisperForAudioClassificationModule(nn.Module):
    config: WhisperConfig
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    def setup(self) -> None:
        self.encoder = FlaxWhisperEncoder(config=self.config, dtype=self.dtype, gradient_checkpointing=self.gradient_checkpointing)
        self.config.is_encoder_decoder = False
        num_layers = self.config.num_hidden_layers + 1
        if self.config.use_weighted_layer_sum:
            self.layer_weights = jnp.repeat(1 / num_layers, num_layers)
        self.projector = nn.Dense(self.config.classifier_proj_size, dtype=self.dtype)
        self.classifier = nn.Dense(self.config.num_labels, dtype=self.dtype)

    def __call__(self, input_features, encoder_outputs=None, output_attentions=None, output_hidden_states: bool=True, return_dict: bool=True):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if encoder_outputs is None:
            encoder_outputs = self.encoder(input_features, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        if self.config.use_weighted_layer_sum:
            hidden_states = jnp.stack(encoder_outputs, axis=1)
            norm_weights = jax.nn.softmax(self.layer_weights, axis=-1)
            hidden_states = jnp.sum(hidden_states * jnp.reshape(norm_weights, [-1, 1, 1]), axis=1)
        else:
            hidden_states = encoder_outputs[0]
        hidden_states = self.projector(hidden_states)
        pooled_output = jnp.mean(hidden_states, axis=1)
        logits = self.classifier(pooled_output)
        if not return_dict:
            return (logits,) + encoder_outputs[1:]
        return FlaxSequenceClassifierOutput(logits=logits, hidden_states=encoder_outputs.hidden_states, attentions=encoder_outputs.attentions)
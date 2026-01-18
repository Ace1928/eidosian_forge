import copy
from typing import Any, Callable, List, Optional, Tuple
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from flax.linen import partitioning as nn_partitioning
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax.random import PRNGKey
from ...modeling_flax_outputs import (
from ...modeling_flax_utils import (
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_longt5 import LongT5Config
@add_start_docstrings('LONGT5 Model with a `language modeling` head on top.', LONGT5_START_DOCSTRING)
class FlaxLongT5ForConditionalGenerationModule(nn.Module):
    config: LongT5Config
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    def _get_encoder_module(self):
        return self.encoder

    def _get_decoder_module(self):
        return self.decoder

    def setup(self):
        self.model_dim = self.config.d_model
        self.shared = nn.Embed(self.config.vocab_size, self.config.d_model, embedding_init=jax.nn.initializers.normal(self.config.initializer_factor), dtype=self.dtype)
        encoder_config = copy.deepcopy(self.config)
        encoder_config.causal = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = FlaxLongT5Stack(encoder_config, self.shared, dtype=self.dtype, gradient_checkpointing=self.gradient_checkpointing)
        decoder_config = copy.deepcopy(self.config)
        decoder_config.causal = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = self.config.num_decoder_layers
        self.decoder = FlaxLongT5Stack(decoder_config, self.shared, dtype=self.dtype, gradient_checkpointing=self.gradient_checkpointing)
        self.lm_head = nn.Dense(self.config.vocab_size, use_bias=False, kernel_init=jax.nn.initializers.normal(self.config.initializer_factor), dtype=self.dtype)

    def __call__(self, input_ids=None, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, encoder_outputs=None, output_attentions=None, output_hidden_states=None, return_dict=None, deterministic: bool=True):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, deterministic=deterministic)
        hidden_states = encoder_outputs[0]
        decoder_outputs = self.decoder(input_ids=decoder_input_ids, attention_mask=decoder_attention_mask, encoder_hidden_states=hidden_states, encoder_attention_mask=attention_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, deterministic=deterministic)
        sequence_output = decoder_outputs[0]
        if self.config.tie_word_embeddings:
            sequence_output = sequence_output * self.model_dim ** (-0.5)
        if self.config.tie_word_embeddings:
            shared_embedding = self.shared.variables['params']['embedding']
            lm_logits = self.lm_head.apply({'params': {'kernel': shared_embedding.T}}, sequence_output)
        else:
            lm_logits = self.lm_head(sequence_output)
        if not return_dict:
            return (lm_logits,) + decoder_outputs[1:] + encoder_outputs
        return FlaxSeq2SeqLMOutput(logits=lm_logits, past_key_values=decoder_outputs.past_key_values, decoder_hidden_states=decoder_outputs.hidden_states, decoder_attentions=decoder_outputs.attentions, cross_attentions=decoder_outputs.cross_attentions, encoder_last_hidden_state=encoder_outputs.last_hidden_state, encoder_hidden_states=encoder_outputs.hidden_states, encoder_attentions=encoder_outputs.attentions)
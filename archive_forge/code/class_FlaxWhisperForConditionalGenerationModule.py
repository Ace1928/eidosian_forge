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
class FlaxWhisperForConditionalGenerationModule(nn.Module):
    config: WhisperConfig
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    def setup(self) -> None:
        self.model = FlaxWhisperModule(config=self.config, dtype=self.dtype, gradient_checkpointing=self.gradient_checkpointing)
        self.lm_head = nn.Dense(self.config.vocab_size, use_bias=False, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(self.config.init_std))

    def _get_encoder_module(self):
        return self.model.encoder

    def _get_decoder_module(self):
        return self.model.decoder

    def __call__(self, input_features, decoder_input_ids, decoder_attention_mask: jnp.ndarray=None, decoder_position_ids: jnp.ndarray=None, position_ids: jnp.ndarray=None, attention_mask: jnp.ndarray=None, output_attentions: bool=False, output_hidden_states: bool=False, return_dict: bool=True, deterministic: bool=True):
        outputs = self.model(input_features=input_features, decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask, decoder_position_ids=decoder_position_ids, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, deterministic=deterministic)
        hidden_states = outputs[0]
        if self.config.tie_word_embeddings:
            shared_embedding = self.model.decoder.embed_tokens.variables['params']['embedding']
            lm_logits = self.lm_head.apply({'params': {'kernel': shared_embedding.T}}, hidden_states)
        else:
            lm_logits = self.lm_head(hidden_states)
        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return output
        return FlaxSeq2SeqLMOutput(logits=lm_logits, decoder_hidden_states=outputs.decoder_hidden_states, decoder_attentions=outputs.decoder_attentions, cross_attentions=outputs.cross_attentions, encoder_last_hidden_state=outputs.encoder_last_hidden_state, encoder_hidden_states=outputs.encoder_hidden_states, encoder_attentions=outputs.encoder_attentions)
import os
from typing import Optional, Tuple, Union
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from jax.random import PRNGKey
from ...modeling_flax_outputs import FlaxBaseModelOutput, FlaxCausalLMOutputWithCrossAttentions, FlaxSeq2SeqLMOutput
from ...modeling_flax_utils import FlaxPreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from ..auto.configuration_auto import AutoConfig
from ..auto.modeling_flax_auto import FlaxAutoModel, FlaxAutoModelForCausalLM
from .configuration_encoder_decoder import EncoderDecoderConfig
class FlaxEncoderDecoderModule(nn.Module):
    config: EncoderDecoderConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        encoder_config = self.config.encoder
        decoder_config = self.config.decoder
        from ...models.auto.modeling_flax_auto import FLAX_MODEL_FOR_CAUSAL_LM_MAPPING, FLAX_MODEL_MAPPING
        encoder_module = FLAX_MODEL_MAPPING[encoder_config.__class__].module_class
        decoder_module = FLAX_MODEL_FOR_CAUSAL_LM_MAPPING[decoder_config.__class__].module_class
        self.encoder = encoder_module(encoder_config, dtype=self.dtype)
        self.decoder = decoder_module(decoder_config, dtype=self.dtype)
        if self.encoder.config.hidden_size != self.decoder.config.hidden_size and self.decoder.config.cross_attention_hidden_size is None:
            self.enc_to_dec_proj = nn.Dense(self.decoder.config.hidden_size, kernel_init=jax.nn.initializers.normal(self.decoder.config.initializer_range), dtype=self.dtype)
        else:
            self.enc_to_dec_proj = None

    def _get_encoder_module(self):
        return self.encoder

    def _get_projection_module(self):
        return self.enc_to_dec_proj

    def _get_decoder_module(self):
        return self.decoder

    def __call__(self, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, position_ids, decoder_position_ids, output_attentions: bool=False, output_hidden_states: bool=False, return_dict: bool=True, deterministic: bool=True):
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, deterministic=deterministic)
        encoder_hidden_states = encoder_outputs[0]
        if self.enc_to_dec_proj is not None:
            encoder_hidden_states = self.enc_to_dec_proj(encoder_hidden_states)
        decoder_outputs = self.decoder(input_ids=decoder_input_ids, attention_mask=decoder_attention_mask, position_ids=decoder_position_ids, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=attention_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, deterministic=deterministic)
        if not return_dict:
            return decoder_outputs + encoder_outputs
        return FlaxSeq2SeqLMOutput(logits=decoder_outputs.logits, decoder_hidden_states=decoder_outputs.hidden_states, decoder_attentions=decoder_outputs.attentions, cross_attentions=decoder_outputs.cross_attentions, encoder_last_hidden_state=encoder_outputs.last_hidden_state, encoder_hidden_states=encoder_outputs.hidden_states, encoder_attentions=encoder_outputs.attentions)
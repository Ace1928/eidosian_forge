import math
import random
from functools import partial
from typing import Callable, Optional, Tuple
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from jax.random import PRNGKey
from ...modeling_flax_outputs import (
from ...modeling_flax_utils import (
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_bart import BartConfig
@add_start_docstrings('The BART Model with a language modeling head. Can be used for summarization.', BART_START_DOCSTRING)
class FlaxBartForConditionalGeneration(FlaxBartPreTrainedModel):
    module_class = FlaxBartForConditionalGenerationModule
    dtype: jnp.dtype = jnp.float32

    @add_start_docstrings(BART_DECODE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=FlaxCausalLMOutputWithCrossAttentions, config_class=BartConfig)
    def decode(self, decoder_input_ids, encoder_outputs, encoder_attention_mask: Optional[jnp.ndarray]=None, decoder_attention_mask: Optional[jnp.ndarray]=None, decoder_position_ids: Optional[jnp.ndarray]=None, past_key_values: dict=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, train: bool=False, params: dict=None, dropout_rng: PRNGKey=None):
        """
        Returns:

        Example:

        ```python
        >>> import jax.numpy as jnp
        >>> from transformers import AutoTokenizer, FlaxBartForConditionalGeneration

        >>> model = FlaxBartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

        >>> text = "My friends are cool but they eat too many carbs."
        >>> inputs = tokenizer(text, max_length=1024, return_tensors="jax")
        >>> encoder_outputs = model.encode(**inputs)

        >>> decoder_start_token_id = model.config.decoder_start_token_id
        >>> decoder_input_ids = jnp.ones((inputs.input_ids.shape[0], 1), dtype="i4") * decoder_start_token_id

        >>> outputs = model.decode(decoder_input_ids, encoder_outputs)
        >>> logits = outputs.logits
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        encoder_hidden_states = encoder_outputs[0]
        if encoder_attention_mask is None:
            batch_size, sequence_length = encoder_hidden_states.shape[:2]
            encoder_attention_mask = jnp.ones((batch_size, sequence_length))
        batch_size, sequence_length = decoder_input_ids.shape
        if decoder_attention_mask is None:
            decoder_attention_mask = jnp.ones((batch_size, sequence_length))
        if decoder_position_ids is None:
            if past_key_values is not None:
                raise ValueError('Make sure to provide `decoder_position_ids` when passing `past_key_values`.')
            decoder_position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))
        rngs = {}
        if dropout_rng is not None:
            rngs['dropout'] = dropout_rng
        inputs = {'params': params or self.params}
        if past_key_values:
            inputs['cache'] = past_key_values
            mutable = ['cache']
        else:
            mutable = False

        def _decoder_forward(module, decoder_input_ids, decoder_attention_mask, decoder_position_ids, **kwargs):
            decoder_module = module._get_decoder_module()
            outputs = decoder_module(decoder_input_ids, decoder_attention_mask, decoder_position_ids, **kwargs)
            hidden_states = outputs[0]
            if self.config.tie_word_embeddings:
                shared_embedding = module.model.variables['params']['shared']['embedding']
                lm_logits = module.lm_head.apply({'params': {'kernel': shared_embedding.T}}, hidden_states)
            else:
                lm_logits = module.lm_head(hidden_states)
            lm_logits += module.final_logits_bias.astype(self.dtype)
            return (lm_logits, outputs)
        outputs = self.module.apply(inputs, decoder_input_ids=jnp.array(decoder_input_ids, dtype='i4'), decoder_attention_mask=jnp.array(decoder_attention_mask, dtype='i4'), decoder_position_ids=jnp.array(decoder_position_ids, dtype='i4'), encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=jnp.array(encoder_attention_mask, dtype='i4'), output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, deterministic=not train, rngs=rngs, mutable=mutable, method=_decoder_forward)
        if past_key_values is None:
            lm_logits, decoder_outputs = outputs
        else:
            (lm_logits, decoder_outputs), past = outputs
        if return_dict:
            outputs = FlaxCausalLMOutputWithCrossAttentions(logits=lm_logits, hidden_states=decoder_outputs.hidden_states, attentions=decoder_outputs.attentions, cross_attentions=decoder_outputs.cross_attentions)
        else:
            outputs = (lm_logits,) + decoder_outputs[1:]
        if past_key_values is not None and return_dict:
            outputs['past_key_values'] = unfreeze(past['cache'])
            return outputs
        elif past_key_values is not None and (not return_dict):
            outputs = outputs[:1] + (unfreeze(past['cache']),) + outputs[1:]
        return outputs

    def prepare_inputs_for_generation(self, decoder_input_ids, max_length, attention_mask: Optional[jax.Array]=None, decoder_attention_mask: Optional[jax.Array]=None, encoder_outputs=None, **kwargs):
        batch_size, seq_length = decoder_input_ids.shape
        past_key_values = self.init_cache(batch_size, max_length, encoder_outputs)
        extended_attention_mask = jnp.ones((batch_size, max_length), dtype='i4')
        if decoder_attention_mask is not None:
            position_ids = decoder_attention_mask.cumsum(axis=-1) - 1
            extended_attention_mask = lax.dynamic_update_slice(extended_attention_mask, decoder_attention_mask, (0, 0))
        else:
            position_ids = jnp.broadcast_to(jnp.arange(seq_length, dtype='i4')[None, :], (batch_size, seq_length))
        return {'past_key_values': past_key_values, 'encoder_outputs': encoder_outputs, 'encoder_attention_mask': attention_mask, 'decoder_attention_mask': extended_attention_mask, 'decoder_position_ids': position_ids}

    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        model_kwargs['past_key_values'] = model_outputs.past_key_values
        model_kwargs['decoder_position_ids'] = model_kwargs['decoder_position_ids'][:, -1:] + 1
        return model_kwargs
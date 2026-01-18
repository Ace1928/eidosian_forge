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
class FlaxWhisperPreTrainedModel(FlaxPreTrainedModel):
    config_class = WhisperConfig
    base_model_prefix: str = 'model'
    main_input_name = 'input_features'
    module_class: nn.Module = None

    def __init__(self, config: WhisperConfig, input_shape: Tuple[int]=None, seed: int=0, dtype: jnp.dtype=jnp.float32, _do_init: bool=True, gradient_checkpointing: bool=False, **kwargs):
        module = self.module_class(config=config, dtype=dtype, gradient_checkpointing=gradient_checkpointing, **kwargs)
        if input_shape is None:
            input_shape = (1, config.num_mel_bins, 2 * config.max_source_positions)
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    def enable_gradient_checkpointing(self):
        self._module = self.module_class(config=self.config, dtype=self.dtype, gradient_checkpointing=True)

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict=None) -> FrozenDict:
        input_features = jnp.zeros(input_shape, dtype='f4')
        input_features = input_features.at[..., -1].set(self.config.eos_token_id)
        decoder_input_ids = jnp.zeros((input_shape[0], 1), dtype='i4')
        decoder_attention_mask = jnp.ones_like(decoder_input_ids)
        batch_size, sequence_length = decoder_input_ids.shape
        decoder_position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {'params': params_rng, 'dropout': dropout_rng}
        random_params = self.module.init(rngs, input_features=input_features, decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask, decoder_position_ids=decoder_position_ids)['params']
        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            return random_params

    def init_cache(self, batch_size, max_length, encoder_outputs):
        """
        Args:
            batch_size (`int`):
                batch_size used for fast auto-regressive decoding. Defines the batch size of the initialized cache.
            max_length (`int`):
                maximum possible length for auto-regressive decoding. Defines the sequence length of the initialized
                cache.
            encoder_outputs (`Union[FlaxBaseModelOutput, tuple(tuple(jnp.ndarray)]`):
                `encoder_outputs` consists of (`last_hidden_state`, *optional*: `hidden_states`, *optional*:
                `attentions`). `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`, *optional*)
                is a sequence of hidden-states at the output of the last layer of the encoder. Used in the
                cross-attention of the decoder.
        """
        decoder_input_ids = jnp.ones((batch_size, max_length), dtype='i4')
        decoder_attention_mask = jnp.ones_like(decoder_input_ids)
        decoder_position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(decoder_input_ids).shape[-1]), decoder_input_ids.shape)

        def _decoder_forward(module, decoder_input_ids, decoder_attention_mask, decoder_position_ids, **kwargs):
            decoder_module = module._get_decoder_module()
            return decoder_module(decoder_input_ids, decoder_attention_mask, decoder_position_ids, **kwargs)
        init_variables = self.module.init(jax.random.PRNGKey(0), decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask, decoder_position_ids=decoder_position_ids, encoder_hidden_states=encoder_outputs[0], init_cache=True, method=_decoder_forward)
        return unfreeze(init_variables['cache'])

    @add_start_docstrings(WHISPER_ENCODE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=FlaxBaseModelOutput, config_class=WhisperConfig)
    def encode(self, input_features: jnp.ndarray, attention_mask: Optional[jnp.ndarray]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, train: bool=False, params: dict=None, dropout_rng: PRNGKey=None, **kwargs):
        """
        Returns:

        Example:

        ```python
        >>> from transformers import WhisperProcessor, FlaxWhisperForConditionalGeneration
        >>> from datasets import load_dataset

        >>> processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
        >>> model = FlaxWhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en", from_pt=True)
        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        >>> inputs = processor(ds[0]["audio"]["array"], return_tensors="np")
        >>> input_features = inputs.input_features
        >>> encoder_outputs = model.encode(input_features=input_features)
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        rngs = {}
        if dropout_rng is not None:
            rngs['dropout'] = dropout_rng

        def _encoder_forward(module, input_features, **kwargs):
            encode_module = module._get_encoder_module()
            return encode_module(input_features, **kwargs)
        return self.module.apply({'params': params or self.params}, input_features=jnp.array(input_features, dtype='f4'), output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, deterministic=not train, rngs=rngs, method=_encoder_forward)

    @add_start_docstrings(WHISPER_DECODE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=FlaxBaseModelOutputWithPastAndCrossAttentions, config_class=WhisperConfig)
    def decode(self, decoder_input_ids, encoder_outputs, encoder_attention_mask: Optional[jnp.ndarray]=None, decoder_attention_mask: Optional[jnp.ndarray]=None, decoder_position_ids: Optional[jnp.ndarray]=None, past_key_values: dict=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, train: bool=False, params: dict=None, dropout_rng: PRNGKey=None):
        """
        Returns:

        Example:

        ```python
        >>> from transformers import WhisperProcessor, FlaxWhisperForConditionalGeneration
        >>> from datasets import load_dataset
        >>> import jax.numpy as jnp

        >>> processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
        >>> model = FlaxWhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en", from_pt=True)
        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        >>> input_features = processor(ds[0]["audio"]["array"], return_tensors="np").input_features

        >>> encoder_outputs = model.encode(input_features=input_features)
        >>> decoder_start_token_id = model.config.decoder_start_token_id

        >>> decoder_input_ids = jnp.ones((input_features.shape[0], 1), dtype="i4") * decoder_start_token_id

        >>> outputs = model.decode(decoder_input_ids, encoder_outputs)
        >>> last_decoder_hidden_states = outputs.last_hidden_state
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        encoder_hidden_states = encoder_outputs[0]
        batch_size, sequence_length = decoder_input_ids.shape
        if decoder_position_ids is None:
            if past_key_values is not None:
                raise ValueError('Make sure to provide `decoder_position_ids` when passing `past_key_values`.')
            if decoder_attention_mask is not None:
                decoder_position_ids = decoder_attention_mask.cumsum(-1) * decoder_attention_mask - 1
            else:
                decoder_position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))
        if decoder_attention_mask is None:
            decoder_attention_mask = jnp.ones((batch_size, sequence_length))
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
            return decoder_module(input_ids=decoder_input_ids, attention_mask=decoder_attention_mask, position_ids=decoder_position_ids, **kwargs)
        outputs = self.module.apply(inputs, decoder_input_ids=jnp.array(decoder_input_ids, dtype='i4'), decoder_attention_mask=jnp.array(decoder_attention_mask, dtype='i4'), decoder_position_ids=jnp.array(decoder_position_ids, dtype='i4'), encoder_hidden_states=encoder_hidden_states, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, deterministic=not train, rngs=rngs, mutable=mutable, method=_decoder_forward)
        if past_key_values is not None and return_dict:
            outputs, past = outputs
            outputs['past_key_values'] = unfreeze(past['cache'])
            return outputs
        elif past_key_values is not None and (not return_dict):
            outputs, past = outputs
            outputs = outputs[:1] + (unfreeze(past['cache']),) + outputs[1:]
        return outputs

    @add_start_docstrings_to_model_forward(WHISPER_INPUTS_DOCSTRING)
    def __call__(self, input_features: jnp.ndarray, decoder_input_ids: jnp.ndarray, attention_mask: Optional[jnp.ndarray]=None, decoder_attention_mask: Optional[jnp.ndarray]=None, position_ids: Optional[jnp.ndarray]=None, decoder_position_ids: Optional[jnp.ndarray]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, train: bool=False, params: dict=None, dropout_rng: PRNGKey=None):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        if decoder_position_ids is None:
            if decoder_attention_mask is not None:
                decoder_position_ids = decoder_attention_mask.cumsum(-1) * decoder_attention_mask - 1
            else:
                batch_size, sequence_length = decoder_input_ids.shape
                decoder_position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))
        if decoder_attention_mask is None:
            decoder_attention_mask = jnp.ones_like(decoder_input_ids)
        rngs = {'dropout': dropout_rng} if dropout_rng is not None else {}
        return self.module.apply({'params': params or self.params}, input_features=jnp.array(input_features, dtype='f4'), decoder_input_ids=jnp.array(decoder_input_ids, dtype='i4'), decoder_attention_mask=jnp.array(decoder_attention_mask, dtype='i4'), decoder_position_ids=jnp.array(decoder_position_ids, dtype='i4'), output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, deterministic=not train, rngs=rngs)
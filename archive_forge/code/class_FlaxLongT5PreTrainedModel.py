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
class FlaxLongT5PreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = LongT5Config
    base_model_prefix = 'transformer'
    module_class: nn.Module = None

    def __init__(self, config: LongT5Config, input_shape: Tuple[int]=(1, 1), seed: int=0, dtype: jnp.dtype=jnp.float32, _do_init: bool=True, **kwargs):
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    def enable_gradient_checkpointing(self):
        self._module = self.module_class(config=self.config, dtype=self.dtype, gradient_checkpointing=True)

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict=None) -> FrozenDict:
        input_ids = jnp.zeros(input_shape, dtype='i4')
        attention_mask = jnp.ones_like(input_ids)
        decoder_input_ids = jnp.ones_like(input_ids)
        decoder_attention_mask = jnp.ones_like(input_ids)
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {'params': params_rng, 'dropout': dropout_rng}
        random_params = self.module.init(rngs, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask)['params']
        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            return random_params

    @add_start_docstrings_to_model_forward(LONGT5_INPUTS_DOCSTRING)
    def __call__(self, input_ids: jnp.ndarray, attention_mask: Optional[jnp.ndarray]=None, decoder_input_ids: jnp.ndarray=None, decoder_attention_mask: Optional[jnp.ndarray]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, train: bool=False, params: dict=None, dropout_rng: PRNGKey=None):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        if decoder_input_ids is None:
            raise ValueError('Make sure to provide both `input_ids` and `decoder_input_ids`. `decoder_input_ids` is not passed here.')
        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)
        if decoder_attention_mask is None:
            decoder_attention_mask = jnp.ones_like(decoder_input_ids)
        rngs = {'dropout': dropout_rng} if dropout_rng is not None else {}
        return self.module.apply({'params': params or self.params}, input_ids=jnp.array(input_ids, dtype='i4'), attention_mask=jnp.array(attention_mask, dtype='i4'), decoder_input_ids=jnp.array(decoder_input_ids, dtype='i4'), decoder_attention_mask=jnp.array(decoder_attention_mask, dtype='i4'), output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, deterministic=not train, rngs=rngs)

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

        def _decoder_forward(module, decoder_input_ids, decoder_attention_mask, **kwargs):
            decoder_module = module._get_decoder_module()
            return decoder_module(decoder_input_ids, decoder_attention_mask, **kwargs)
        init_variables = self.module.init(jax.random.PRNGKey(0), decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask, encoder_hidden_states=encoder_outputs[0], init_cache=True, method=_decoder_forward)
        return unfreeze(init_variables['cache'])

    @add_start_docstrings(LONGT5_ENCODE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=FlaxBaseModelOutput, config_class=LongT5Config)
    def encode(self, input_ids: jnp.ndarray, attention_mask: Optional[jnp.ndarray]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, train: bool=False, params: dict=None, dropout_rng: PRNGKey=None):
        """
        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, FlaxLongT5ForConditionalGeneration

        >>> tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base")
        >>> model = FlaxLongT5ForConditionalGeneration.from_pretrained("google/long-t5-local-base")

        >>> text = "My friends are cool but they eat too many carbs."
        >>> inputs = tokenizer(text, return_tensors="np")
        >>> encoder_outputs = model.encode(**inputs)
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)
        rngs = {}
        if dropout_rng is not None:
            rngs['dropout'] = dropout_rng

        def _encoder_forward(module, input_ids, attention_mask, **kwargs):
            encode_module = module._get_encoder_module()
            return encode_module(input_ids, attention_mask, **kwargs)
        return self.module.apply({'params': params or self.params}, input_ids=jnp.array(input_ids, dtype='i4'), attention_mask=jnp.array(attention_mask, dtype='i4'), output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, deterministic=not train, rngs=rngs, method=_encoder_forward)

    @add_start_docstrings(LONGT5_DECODE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=FlaxBaseModelOutputWithPastAndCrossAttentions, config_class=LongT5Config)
    def decode(self, decoder_input_ids, encoder_outputs, encoder_attention_mask: Optional[jnp.ndarray]=None, decoder_attention_mask: Optional[jnp.ndarray]=None, past_key_values: dict=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, train: bool=False, params: dict=None, dropout_rng: PRNGKey=None):
        """
        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, FlaxLongT5ForConditionalGeneration
        >>> import jax.numpy as jnp

        >>> tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base")
        >>> model = FlaxLongT5ForConditionalGeneration.from_pretrained("google/long-t5-local-base")

        >>> text = "My friends are cool but they eat too many carbs."
        >>> inputs = tokenizer(text, return_tensors="np")
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
        rngs = {}
        if dropout_rng is not None:
            rngs['dropout'] = dropout_rng
        inputs = {'params': params or self.params}
        if past_key_values:
            inputs['cache'] = past_key_values
            mutable = ['cache']
        else:
            mutable = False

        def _decoder_forward(module, decoder_input_ids, decoder_attention_mask, **kwargs):
            decoder_module = module._get_decoder_module()
            return decoder_module(decoder_input_ids, decoder_attention_mask, **kwargs)
        outputs = self.module.apply(inputs, decoder_input_ids=jnp.array(decoder_input_ids, dtype='i4'), decoder_attention_mask=jnp.array(decoder_attention_mask, dtype='i4'), encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=jnp.array(encoder_attention_mask, dtype='i4'), output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, deterministic=not train, rngs=rngs, mutable=mutable, method=_decoder_forward)
        if past_key_values is not None and return_dict:
            outputs, past = outputs
            outputs['past_key_values'] = unfreeze(past['cache'])
            return outputs
        elif past_key_values is not None and (not return_dict):
            outputs, past = outputs
            outputs = outputs[:1] + (unfreeze(past['cache']),) + outputs[1:]
        return outputs
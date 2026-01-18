from typing import Optional, Tuple
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from ...modeling_flax_outputs import (
from ...modeling_flax_utils import ACT2FN, FlaxPreTrainedModel, append_call_sample_docstring, logging
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward
from .configuration_mistral import MistralConfig
class FlaxMistralPreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = MistralConfig
    base_model_prefix = 'model'
    module_class: nn.Module = None

    def __init__(self, config: MistralConfig, input_shape: Tuple=(1, 1), seed: int=0, dtype: jnp.dtype=jnp.float32, _do_init: bool=True, **kwargs):
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict=None) -> FrozenDict:
        input_ids = jnp.zeros(input_shape, dtype='i4')
        attention_mask = jnp.ones_like(input_ids)
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_shape)
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {'params': params_rng, 'dropout': dropout_rng}
        random_params = self.module.init(rngs, input_ids, attention_mask, position_ids, return_dict=False)['params']
        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            return random_params

    def init_cache(self, batch_size, max_length):
        """
        Args:
            batch_size (`int`):
                batch_size used for fast auto-regressive decoding. Defines the batch size of the initialized cache.
            max_length (`int`):
                maximum possible length for auto-regressive decoding. Defines the sequence length of the initialized
                cache.
        """
        input_ids = jnp.ones((batch_size, max_length))
        attention_mask = jnp.ones_like(input_ids)
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)
        init_variables = self.module.init(jax.random.PRNGKey(0), input_ids, attention_mask, position_ids, return_dict=False, init_cache=True)
        return unfreeze(init_variables['cache'])

    @add_start_docstrings_to_model_forward(MISTRAL_INPUTS_DOCSTRING)
    def __call__(self, input_ids, attention_mask=None, position_ids=None, params: dict=None, past_key_values: dict=None, dropout_rng: jax.random.PRNGKey=None, train: bool=False, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        batch_size, sequence_length = input_ids.shape
        if position_ids is None:
            if past_key_values is not None:
                raise ValueError('Make sure to provide `position_ids` when passing `past_key_values`.')
            position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))
        if attention_mask is None:
            attention_mask = jnp.ones((batch_size, sequence_length))
        rngs = {}
        if dropout_rng is not None:
            rngs['dropout'] = dropout_rng
        inputs = {'params': params or self.params}
        if past_key_values:
            inputs['cache'] = past_key_values
            mutable = ['cache']
        else:
            mutable = False
        outputs = self.module.apply(inputs, jnp.array(input_ids, dtype='i4'), jnp.array(attention_mask, dtype='i4'), jnp.array(position_ids, dtype='i4'), not train, False, output_attentions, output_hidden_states, return_dict, rngs=rngs, mutable=mutable)
        if past_key_values is not None and return_dict:
            outputs, past_key_values = outputs
            outputs['past_key_values'] = unfreeze(past_key_values['cache'])
            return outputs
        elif past_key_values is not None and (not return_dict):
            outputs, past_key_values = outputs
            outputs = outputs[:1] + (unfreeze(past_key_values['cache']),) + outputs[1:]
        return outputs
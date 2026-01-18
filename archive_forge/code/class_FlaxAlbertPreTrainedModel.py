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
class FlaxAlbertPreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = AlbertConfig
    base_model_prefix = 'albert'
    module_class: nn.Module = None

    def __init__(self, config: AlbertConfig, input_shape: Tuple=(1, 1), seed: int=0, dtype: jnp.dtype=jnp.float32, _do_init: bool=True, **kwargs):
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict=None) -> FrozenDict:
        input_ids = jnp.zeros(input_shape, dtype='i4')
        token_type_ids = jnp.zeros_like(input_ids)
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_shape)
        attention_mask = jnp.ones_like(input_ids)
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {'params': params_rng, 'dropout': dropout_rng}
        random_params = self.module.init(rngs, input_ids, attention_mask, token_type_ids, position_ids, return_dict=False)['params']
        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            return random_params

    @add_start_docstrings_to_model_forward(ALBERT_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    def __call__(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, params: dict=None, dropout_rng: jax.random.PRNGKey=None, train: bool=False, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        if token_type_ids is None:
            token_type_ids = jnp.zeros_like(input_ids)
        if position_ids is None:
            position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)
        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)
        rngs = {}
        if dropout_rng is not None:
            rngs['dropout'] = dropout_rng
        return self.module.apply({'params': params or self.params}, jnp.array(input_ids, dtype='i4'), jnp.array(attention_mask, dtype='i4'), jnp.array(token_type_ids, dtype='i4'), jnp.array(position_ids, dtype='i4'), not train, output_attentions, output_hidden_states, return_dict, rngs=rngs)
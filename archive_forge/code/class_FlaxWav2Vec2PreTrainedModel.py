from functools import partial
from typing import Optional, Tuple, Union
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from ...modeling_flax_outputs import FlaxBaseModelOutput, FlaxCausalLMOutput
from ...modeling_flax_utils import (
from ...utils import ModelOutput, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_wav2vec2 import Wav2Vec2Config
class FlaxWav2Vec2PreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = Wav2Vec2Config
    base_model_prefix: str = 'wav2vec2'
    main_input_name = 'input_values'
    module_class: nn.Module = None

    def __init__(self, config: Wav2Vec2Config, input_shape: Tuple=(1, 1024), seed: int=0, dtype: jnp.dtype=jnp.float32, _do_init: bool=True, **kwargs):
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict=None) -> FrozenDict:
        input_values = jnp.zeros(input_shape, dtype='i4')
        attention_mask = jnp.ones_like(input_values)
        params_rng, dropout_rng = jax.random.split(rng, 2)
        rngs = {'params': params_rng, 'dropout': dropout_rng}
        random_params = self.module.init(rngs, input_values, attention_mask, return_dict=False)['params']
        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            return random_params

    @add_start_docstrings_to_model_forward(WAV_2_VEC_2_INPUTS_DOCSTRING)
    def __call__(self, input_values, attention_mask=None, mask_time_indices=None, params: dict=None, dropout_rng: jax.random.PRNGKey=None, train: bool=False, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, freeze_feature_encoder: bool=False, return_dict: Optional[bool]=None):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        batch_size, sequence_length = input_values.shape
        if attention_mask is None:
            attention_mask = jnp.ones((batch_size, sequence_length))
        rngs = {}
        if dropout_rng is not None:
            rngs['dropout'] = dropout_rng
        inputs = {'params': params or self.params}
        return self.module.apply(inputs, jnp.array(input_values, dtype='f4'), jnp.array(attention_mask, dtype='i4'), mask_time_indices, not train, output_attentions, output_hidden_states, freeze_feature_encoder, return_dict, rngs=rngs)

    def _get_feat_extract_output_lengths(self, input_lengths: Union[jnp.ndarray, int], add_adapter: Optional[bool]=None):
        return self.module._get_feat_extract_output_lengths(input_lengths, add_adapter=add_adapter)
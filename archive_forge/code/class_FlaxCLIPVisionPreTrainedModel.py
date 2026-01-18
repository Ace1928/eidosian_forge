from typing import Any, Optional, Tuple, Union
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from ...modeling_flax_outputs import FlaxBaseModelOutput, FlaxBaseModelOutputWithPooling
from ...modeling_flax_utils import (
from ...utils import ModelOutput, add_start_docstrings, logging
from .configuration_clip import CLIPConfig, CLIPTextConfig, CLIPVisionConfig
class FlaxCLIPVisionPreTrainedModel(FlaxPreTrainedModel):
    config_class = CLIPVisionConfig
    main_input_name = 'pixel_values'
    module_class: nn.Module = None

    def __init__(self, config: CLIPVisionConfig, input_shape: Optional[Tuple]=None, seed: int=0, dtype: jnp.dtype=jnp.float32, _do_init: bool=True, **kwargs):
        if input_shape is None:
            input_shape = (1, config.image_size, config.image_size, 3)
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict=None) -> FrozenDict:
        pixel_values = jax.random.normal(rng, input_shape)
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {'params': params_rng, 'dropout': dropout_rng}
        random_params = self.module.init(rngs, pixel_values)['params']
        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            return random_params

    def __call__(self, pixel_values, params: dict=None, dropout_rng: jax.random.PRNGKey=None, train: bool=False, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        pixel_values = jnp.transpose(pixel_values, (0, 2, 3, 1))
        rngs = {}
        if dropout_rng is not None:
            rngs['dropout'] = dropout_rng
        return self.module.apply({'params': params or self.params}, jnp.array(pixel_values, dtype=jnp.float32), not train, output_attentions, output_hidden_states, return_dict, rngs=rngs)
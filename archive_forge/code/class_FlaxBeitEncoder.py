from typing import Callable, List, Optional, Tuple
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from ...modeling_flax_outputs import (
from ...modeling_flax_utils import (
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward
from .configuration_beit import BeitConfig
class FlaxBeitEncoder(nn.Module):
    config: BeitConfig
    window_size: Tuple[int, int]
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if self.config.use_shared_relative_position_bias:
            self.relative_position_bias = FlaxBeitRelativePositionBias(config=self.config, window_size=self.window_size, dtype=self.dtype)
        drop_path_rates = list(np.linspace(0, self.config.drop_path_rate, self.config.num_hidden_layers))
        self.layer = FlaxBeitLayerCollection(self.config, window_size=self.window_size, drop_path_rates=drop_path_rates, relative_position_bias=self.relative_position_bias if self.config.use_shared_relative_position_bias else None, dtype=self.dtype)

    def __call__(self, hidden_states, deterministic: bool=True, output_attentions: bool=False, output_hidden_states: bool=False, return_dict: bool=True):
        return self.layer(hidden_states, deterministic=deterministic, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
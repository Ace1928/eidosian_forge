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
class FlaxBeitLayer(nn.Module):
    config: BeitConfig
    window_size: Tuple[int, int]
    drop_path_rate: float
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.attention = FlaxBeitAttention(self.config, self.window_size, dtype=self.dtype)
        self.intermediate = FlaxBeitIntermediate(self.config, dtype=self.dtype)
        self.output = FlaxBeitOutput(self.config, dtype=self.dtype)
        self.layernorm_before = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        self.drop_path = FlaxBeitDropPath(rate=self.drop_path_rate)
        self.layernorm_after = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        self.init_values = self.config.layer_scale_init_value
        if self.init_values > 0:
            self.lambda_1 = self.param('lambda_1', ones_with_scale, self.config.hidden_size, self.init_values)
            self.lambda_2 = self.param('lambda_2', ones_with_scale, self.config.hidden_size, self.init_values)
        else:
            self.lambda_1 = None
            self.lambda_2 = None

    def __call__(self, hidden_states, relative_position_bias=None, deterministic: bool=True, output_attentions: bool=False):
        self_attention_outputs = self.attention(self.layernorm_before(hidden_states), relative_position_bias, deterministic=deterministic, output_attentions=output_attentions)
        attention_output = self_attention_outputs[0]
        if self.lambda_1 is not None:
            attention_output = self.lambda_1.astype(attention_output.dtype) * attention_output
        hidden_states = self.drop_path(attention_output, deterministic=deterministic) + hidden_states
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)
        layer_output = self.output(layer_output, deterministic=deterministic)
        if self.lambda_2 is not None:
            layer_output = self.lambda_2.astype(layer_output.dtype) * layer_output
        layer_output = self.drop_path(layer_output, deterministic=deterministic) + hidden_states
        outputs = (layer_output,)
        if output_attentions:
            outputs += (self_attention_outputs[1],)
        return outputs
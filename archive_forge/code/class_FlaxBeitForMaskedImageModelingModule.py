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
class FlaxBeitForMaskedImageModelingModule(nn.Module):
    config: BeitConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.beit = FlaxBeitModule(self.config, add_pooling_layer=False, dtype=self.dtype)
        self.layernorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        self.lm_head = nn.Dense(self.config.vocab_size, kernel_init=jax.nn.initializers.normal(self.config.initializer_range), dtype=self.dtype)

    def __call__(self, pixel_values=None, bool_masked_pos=None, deterministic: bool=True, output_attentions=None, output_hidden_states=None, return_dict=None):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.beit(pixel_values, bool_masked_pos, deterministic=deterministic, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = outputs[0]
        sequence_output = self.layernorm(sequence_output)
        prediction_scores = self.lm_head(sequence_output[:, 1:])
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return output
        return FlaxMaskedLMOutput(logits=prediction_scores, hidden_states=outputs.hidden_states, attentions=outputs.attentions)
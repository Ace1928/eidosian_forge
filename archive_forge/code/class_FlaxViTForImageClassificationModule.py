from typing import Optional, Tuple
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from ...modeling_flax_outputs import FlaxBaseModelOutput, FlaxBaseModelOutputWithPooling, FlaxSequenceClassifierOutput
from ...modeling_flax_utils import (
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward
from .configuration_vit import ViTConfig
class FlaxViTForImageClassificationModule(nn.Module):
    config: ViTConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.vit = FlaxViTModule(config=self.config, dtype=self.dtype, add_pooling_layer=False)
        self.classifier = nn.Dense(self.config.num_labels, dtype=self.dtype, kernel_init=jax.nn.initializers.variance_scaling(self.config.initializer_range ** 2, 'fan_in', 'truncated_normal'))

    def __call__(self, pixel_values=None, deterministic: bool=True, output_attentions=None, output_hidden_states=None, return_dict=None):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.vit(pixel_values, deterministic=deterministic, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        hidden_states = outputs[0]
        logits = self.classifier(hidden_states[:, 0, :])
        if not return_dict:
            output = (logits,) + outputs[2:]
            return output
        return FlaxSequenceClassifierOutput(logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)
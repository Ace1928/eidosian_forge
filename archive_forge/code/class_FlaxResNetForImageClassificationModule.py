from functools import partial
from typing import Optional, Tuple
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict
from ...modeling_flax_outputs import (
from ...modeling_flax_utils import (
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward
from .configuration_resnet import ResNetConfig
class FlaxResNetForImageClassificationModule(nn.Module):
    config: ResNetConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.resnet = FlaxResNetModule(config=self.config, dtype=self.dtype)
        if self.config.num_labels > 0:
            self.classifier = FlaxResNetClassifierCollection(self.config, dtype=self.dtype)
        else:
            self.classifier = Identity()

    def __call__(self, pixel_values=None, deterministic: bool=True, output_hidden_states=None, return_dict=None):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.resnet(pixel_values, deterministic=deterministic, output_hidden_states=output_hidden_states, return_dict=return_dict)
        pooled_output = outputs.pooler_output if return_dict else outputs[1]
        logits = self.classifier(pooled_output[:, :, 0, 0])
        if not return_dict:
            output = (logits,) + outputs[2:]
            return output
        return FlaxImageClassifierOutputWithNoAttention(logits=logits, hidden_states=outputs.hidden_states)
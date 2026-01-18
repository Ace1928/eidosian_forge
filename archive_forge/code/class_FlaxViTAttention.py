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
class FlaxViTAttention(nn.Module):
    config: ViTConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.attention = FlaxViTSelfAttention(self.config, dtype=self.dtype)
        self.output = FlaxViTSelfOutput(self.config, dtype=self.dtype)

    def __call__(self, hidden_states, deterministic=True, output_attentions: bool=False):
        attn_outputs = self.attention(hidden_states, deterministic=deterministic, output_attentions=output_attentions)
        attn_output = attn_outputs[0]
        hidden_states = self.output(attn_output, hidden_states, deterministic=deterministic)
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_outputs[1],)
        return outputs
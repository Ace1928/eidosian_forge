from typing import Callable, Optional, Tuple
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from ...modeling_flax_outputs import (
from ...modeling_flax_utils import ACT2FN, FlaxPreTrainedModel, append_call_sample_docstring, overwrite_call_docstring
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_roformer import RoFormerConfig
class FlaxRoFormerLayerCollection(nn.Module):
    config: RoFormerConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.layers = [FlaxRoFormerLayer(self.config, name=str(i), dtype=self.dtype) for i in range(self.config.num_hidden_layers)]

    def __call__(self, hidden_states, attention_mask, sinusoidal_pos, head_mask, deterministic: bool=True, output_attentions: bool=False, output_hidden_states: bool=False, return_dict: bool=True):
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        if head_mask is not None:
            if head_mask.shape[0] != len(self.layers):
                raise ValueError(f'The head_mask should be specified for {len(self.layers)} layers, but it is for                         {head_mask.shape[0]}.')
        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            layer_outputs = layer(hidden_states, attention_mask, sinusoidal_pos, layer_head_mask=head_mask[i] if head_mask is not None else None, deterministic=deterministic, output_attentions=output_attentions)
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions += (layer_outputs[1],)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        outputs = (hidden_states,)
        if not return_dict:
            return tuple((v for v in outputs if v is not None))
        return FlaxBaseModelOutput(last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions)
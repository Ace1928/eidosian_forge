from functools import partial
from typing import Optional, Tuple
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from ...modeling_flax_outputs import FlaxBaseModelOutput, FlaxCausalLMOutput
from ...modeling_flax_utils import ACT2FN, FlaxPreTrainedModel, append_call_sample_docstring
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_gptj import GPTJConfig
class FlaxGPTJBlockCollection(nn.Module):
    config: GPTJConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.blocks = [FlaxGPTJBlock(self.config, name=str(i), dtype=self.dtype) for i in range(self.config.num_hidden_layers)]

    def __call__(self, hidden_states, attention_mask=None, position_ids=None, deterministic: bool=True, init_cache: bool=False, output_attentions: bool=False, output_hidden_states: bool=False, return_dict: bool=True):
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        for block in self.blocks:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            layer_outputs = block(hidden_states, attention_mask, position_ids=position_ids, deterministic=deterministic, init_cache=init_cache, output_attentions=output_attentions)
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions += (layer_outputs[1],)
        outputs = (hidden_states, all_hidden_states, all_attentions)
        return outputs
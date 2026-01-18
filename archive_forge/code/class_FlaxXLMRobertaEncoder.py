from typing import Callable, Optional, Tuple
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from flax.linen import partitioning as nn_partitioning
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from ...modeling_flax_outputs import (
from ...modeling_flax_utils import ACT2FN, FlaxPreTrainedModel, append_call_sample_docstring, overwrite_call_docstring
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_xlm_roberta import XLMRobertaConfig
class FlaxXLMRobertaEncoder(nn.Module):
    config: XLMRobertaConfig
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    def setup(self):
        self.layer = FlaxXLMRobertaLayerCollection(self.config, dtype=self.dtype, gradient_checkpointing=self.gradient_checkpointing)

    def __call__(self, hidden_states, attention_mask, head_mask, encoder_hidden_states: Optional[jnp.ndarray]=None, encoder_attention_mask: Optional[jnp.ndarray]=None, init_cache: bool=False, deterministic: bool=True, output_attentions: bool=False, output_hidden_states: bool=False, return_dict: bool=True):
        return self.layer(hidden_states, attention_mask, head_mask=head_mask, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask, init_cache=init_cache, deterministic=deterministic, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
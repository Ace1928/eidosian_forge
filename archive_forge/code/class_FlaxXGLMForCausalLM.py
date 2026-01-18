import math
import random
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
from jax.random import PRNGKey
from ...modeling_flax_outputs import (
from ...modeling_flax_utils import ACT2FN, FlaxPreTrainedModel, append_call_sample_docstring
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_xglm import XGLMConfig
@add_start_docstrings('\n    The XGLM Model transformer with a language modeling head on top (linear layer with weights tied to the input\n    embeddings).\n    ', XGLM_START_DOCSTRING)
class FlaxXGLMForCausalLM(FlaxXGLMPreTrainedModel):
    module_class = FlaxXGLMForCausalLMModule

    def prepare_inputs_for_generation(self, input_ids, max_length, attention_mask: Optional[jax.Array]=None):
        batch_size, seq_length = input_ids.shape
        past_key_values = self.init_cache(batch_size, max_length)
        extended_attention_mask = jnp.ones((batch_size, max_length), dtype='i4')
        if attention_mask is not None:
            position_ids = attention_mask.cumsum(axis=-1) - 1
            extended_attention_mask = lax.dynamic_update_slice(extended_attention_mask, attention_mask, (0, 0))
        else:
            position_ids = jnp.broadcast_to(jnp.arange(seq_length, dtype='i4')[None, :], (batch_size, seq_length))
        return {'past_key_values': past_key_values, 'attention_mask': extended_attention_mask, 'position_ids': position_ids}

    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        model_kwargs['past_key_values'] = model_outputs.past_key_values
        model_kwargs['position_ids'] = model_kwargs['position_ids'][:, -1:] + 1
        return model_kwargs
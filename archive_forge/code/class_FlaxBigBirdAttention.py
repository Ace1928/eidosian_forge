from typing import Callable, Optional, Tuple
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from flax.linen import partitioning as nn_partitioning
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from ...modeling_flax_outputs import (
from ...modeling_flax_utils import (
from ...utils import ModelOutput, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_big_bird import BigBirdConfig
class FlaxBigBirdAttention(nn.Module):
    config: BigBirdConfig
    layer_id: int = None
    causal: bool = False
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if self.config.attention_type == 'original_full':
            self.self = FlaxBigBirdSelfAttention(self.config, causal=self.causal, dtype=self.dtype)
        elif self.config.attention_type == 'block_sparse':
            self.self = FlaxBigBirdBlockSparseAttention(self.config, block_sparse_seed=self.layer_id, dtype=self.dtype)
        else:
            raise ValueError(f'Your `config.attention_type` is {self.config.attention_type} but it can either be `original_full` or `block_sparse`')
        self.output = FlaxBigBirdSelfOutput(self.config, dtype=self.dtype)

    def __call__(self, hidden_states, attention_mask, layer_head_mask, key_value_states=None, init_cache=False, deterministic=True, output_attentions: bool=False):
        if self.config.attention_type == 'original_full':
            attn_outputs = self.self(hidden_states, attention_mask, layer_head_mask=layer_head_mask, key_value_states=key_value_states, init_cache=init_cache, deterministic=deterministic, output_attentions=output_attentions)
        else:
            attn_outputs = self.self(hidden_states, attention_mask, deterministic=deterministic, output_attentions=output_attentions)
        attn_output = attn_outputs[0]
        hidden_states = self.output(attn_output, hidden_states, deterministic=deterministic)
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_outputs[1],)
        return outputs
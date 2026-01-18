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
class FlaxBigBirdModule(nn.Module):
    config: BigBirdConfig
    dtype: jnp.dtype = jnp.float32
    add_pooling_layer: bool = True
    gradient_checkpointing: bool = False

    def setup(self):
        self.embeddings = FlaxBigBirdEmbeddings(self.config, dtype=self.dtype)
        self.encoder = FlaxBigBirdEncoder(self.config, dtype=self.dtype, gradient_checkpointing=self.gradient_checkpointing)
        self.pooler = nn.Dense(self.config.hidden_size, kernel_init=jax.nn.initializers.normal(self.config.initializer_range), dtype=self.dtype)

    def __call__(self, input_ids, attention_mask, token_type_ids, position_ids, head_mask, encoder_hidden_states: Optional[jnp.ndarray]=None, encoder_attention_mask: Optional[jnp.ndarray]=None, init_cache: bool=False, deterministic: bool=True, output_attentions: bool=False, output_hidden_states: bool=False, return_dict: bool=True):
        hidden_states = self.embeddings(input_ids, token_type_ids, position_ids, attention_mask, deterministic=deterministic)
        outputs = self.encoder(hidden_states, attention_mask, head_mask=head_mask, deterministic=deterministic, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask, init_cache=init_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        hidden_states = outputs[0]
        pooled = nn.tanh(self.pooler(hidden_states[:, 0, :])) if self.add_pooling_layer else None
        if not return_dict:
            if pooled is None:
                return (hidden_states,) + outputs[1:]
            return (hidden_states, pooled) + outputs[1:]
        return FlaxBaseModelOutputWithPoolingAndCrossAttentions(last_hidden_state=hidden_states, pooler_output=pooled, hidden_states=outputs.hidden_states, attentions=outputs.attentions, cross_attentions=outputs.cross_attentions)
from typing import Any, Optional, Tuple
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from ...modeling_flax_outputs import (
from ...modeling_flax_utils import ACT2FN, FlaxPreTrainedModel, append_call_sample_docstring
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_gpt2 import GPT2Config
class FlaxGPT2Module(nn.Module):
    config: GPT2Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.embed_dim = self.config.hidden_size
        self.wte = nn.Embed(self.config.vocab_size, self.embed_dim, embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range), dtype=self.dtype)
        self.wpe = nn.Embed(self.config.max_position_embeddings, self.embed_dim, embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range), dtype=self.dtype)
        self.dropout = nn.Dropout(rate=self.config.embd_pdrop)
        self.h = FlaxGPT2BlockCollection(self.config, dtype=self.dtype)
        self.ln_f = nn.LayerNorm(epsilon=self.config.layer_norm_epsilon, dtype=self.dtype)

    def __call__(self, input_ids, attention_mask, position_ids, encoder_hidden_states: Optional[jnp.ndarray]=None, encoder_attention_mask: Optional[jnp.ndarray]=None, deterministic=True, init_cache: bool=False, output_attentions: bool=False, output_hidden_states: bool=False, return_dict: bool=True):
        input_embeds = self.wte(input_ids.astype('i4'))
        position_embeds = self.wpe(position_ids.astype('i4'))
        hidden_states = input_embeds + position_embeds
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        outputs = self.h(hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, deterministic=deterministic, init_cache=init_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        hidden_states = outputs[0]
        hidden_states = self.ln_f(hidden_states)
        if output_hidden_states:
            all_hidden_states = outputs[1] + (hidden_states,)
            outputs = (hidden_states, all_hidden_states) + outputs[2:]
        else:
            outputs = (hidden_states,) + outputs[1:]
        if not return_dict:
            return tuple((v for v in outputs if v is not None))
        return FlaxBaseModelOutputWithPastAndCrossAttentions(last_hidden_state=hidden_states, hidden_states=outputs[1], attentions=outputs[2], cross_attentions=outputs[3])
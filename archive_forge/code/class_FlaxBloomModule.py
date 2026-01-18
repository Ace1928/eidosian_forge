import math
from functools import partial
from typing import Optional, Tuple
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, dot_product_attention_weights, make_causal_mask
from flax.linen.activation import tanh
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from ...modeling_flax_outputs import (
from ...modeling_flax_utils import FlaxPreTrainedModel, append_call_sample_docstring
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_bloom import BloomConfig
class FlaxBloomModule(nn.Module):
    config: BloomConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.embed_dim = self.config.hidden_size
        self.word_embeddings = nn.Embed(self.config.vocab_size, self.embed_dim, embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range), dtype=self.dtype)
        self.word_embeddings_layernorm = nn.LayerNorm(epsilon=self.config.layer_norm_epsilon, dtype=self.dtype)
        self.h = FlaxBloomBlockCollection(self.config, dtype=self.dtype)
        self.ln_f = nn.LayerNorm(epsilon=self.config.layer_norm_epsilon, dtype=self.dtype)

    def __call__(self, input_ids=None, attention_mask=None, deterministic=True, init_cache: bool=False, output_attentions: bool=False, output_hidden_states: bool=False, return_dict: bool=True):
        inputs_embeds = self.word_embeddings(input_ids)
        hidden_states = self.word_embeddings_layernorm(inputs_embeds)
        alibi = build_alibi_tensor(attention_mask, self.config.n_head, dtype=hidden_states.dtype)
        outputs = self.h(hidden_states, alibi=alibi, attention_mask=attention_mask, deterministic=deterministic, init_cache=init_cache, output_hidden_states=output_hidden_states, output_attentions=output_attentions)
        hidden_states = outputs[0]
        hidden_states = self.ln_f(hidden_states)
        if output_hidden_states:
            all_hidden_states = outputs[1] + (hidden_states,)
            outputs = (hidden_states, all_hidden_states) + outputs[2:]
        else:
            outputs = (hidden_states,) + outputs[1:]
        if not return_dict:
            return tuple((v for v in [outputs[0], outputs[-1]] if v is not None))
        return FlaxBaseModelOutputWithPastAndCrossAttentions(last_hidden_state=hidden_states, hidden_states=outputs[1], attentions=outputs[-1])
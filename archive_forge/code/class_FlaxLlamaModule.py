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
from .configuration_llama import LlamaConfig
class FlaxLlamaModule(nn.Module):
    config: LlamaConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.hidden_size = self.config.hidden_size
        embedding_init = jax.nn.initializers.normal(stddev=self.config.initializer_range)
        self.embed_tokens = nn.Embed(self.config.vocab_size, self.hidden_size, embedding_init=embedding_init, dtype=self.dtype)
        self.layers = FlaxLlamaLayerCollection(self.config, dtype=self.dtype)
        self.norm = FlaxLlamaRMSNorm(self.config, dtype=self.dtype)

    def __call__(self, input_ids, attention_mask=None, position_ids=None, deterministic=True, init_cache: bool=False, output_attentions: bool=False, output_hidden_states: bool=False, return_dict: bool=True):
        input_embeds = self.embed_tokens(input_ids.astype('i4'))
        outputs = self.layers(input_embeds, position_ids=position_ids, attention_mask=attention_mask, deterministic=deterministic, init_cache=init_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        hidden_states = outputs[0]
        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states = outputs[1] + (hidden_states,)
            outputs = (hidden_states, all_hidden_states) + outputs[2:]
        else:
            outputs = (hidden_states,) + outputs[1:]
        if not return_dict:
            return tuple((v for v in outputs if v is not None))
        return FlaxBaseModelOutput(last_hidden_state=hidden_states, hidden_states=outputs[1], attentions=outputs[-1])
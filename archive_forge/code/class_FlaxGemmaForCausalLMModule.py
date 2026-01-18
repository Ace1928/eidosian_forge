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
from .configuration_gemma import GemmaConfig
class FlaxGemmaForCausalLMModule(nn.Module):
    config: GemmaConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.model = FlaxGemmaModule(self.config, dtype=self.dtype)
        self.lm_head = nn.Dense(self.config.vocab_size, use_bias=False, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range))

    def __call__(self, input_ids, attention_mask=None, position_ids=None, deterministic: bool=True, init_cache: bool=False, output_attentions: bool=False, output_hidden_states: bool=False, return_dict: bool=True):
        outputs = self.model(input_ids, position_ids=position_ids, attention_mask=attention_mask, deterministic=deterministic, init_cache=init_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        hidden_states = outputs[0]
        if self.config.tie_word_embeddings:
            shared_kernel = self.model.variables['params']['embed_tokens']['embedding'].T
            lm_logits = self.lm_head.apply({'params': {'kernel': shared_kernel}}, hidden_states)
        else:
            lm_logits = self.lm_head(hidden_states)
        if not return_dict:
            return (lm_logits,) + outputs[1:]
        return FlaxCausalLMOutput(logits=lm_logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)
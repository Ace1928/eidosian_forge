from functools import partial
from typing import Optional, Tuple, Union
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax
from ...modeling_flax_outputs import FlaxBaseModelOutput, FlaxCausalLMOutput
from ...modeling_flax_utils import (
from ...utils import ModelOutput, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_wav2vec2 import Wav2Vec2Config
class FlaxConvLayersCollection(nn.Module):
    config: Wav2Vec2Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        if self.config.feat_extract_norm == 'layer':
            self.layers = [FlaxWav2Vec2LayerNormConvLayer(self.config, layer_id=i, name=str(i), dtype=self.dtype) for i in range(self.config.num_feat_extract_layers)]
        elif self.config.feat_extract_norm == 'group':
            raise NotImplementedError("At the moment only ``config.feat_extact_norm == 'layer'`` is supported")
        else:
            raise ValueError(f"`config.feat_extract_norm` is {self.config.feat_extract_norm}, but has to be one of ['group', 'layer']")

    def __call__(self, hidden_states):
        for i, conv_layer in enumerate(self.layers):
            hidden_states = conv_layer(hidden_states)
        return hidden_states
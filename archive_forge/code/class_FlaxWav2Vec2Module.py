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
class FlaxWav2Vec2Module(nn.Module):
    config: Wav2Vec2Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.feature_extractor = FlaxWav2Vec2FeatureEncoder(self.config, dtype=self.dtype)
        self.feature_projection = FlaxWav2Vec2FeatureProjection(self.config, dtype=self.dtype)
        self.masked_spec_embed = self.param('masked_spec_embed', jax.nn.initializers.uniform(), (self.config.hidden_size,))
        if self.config.do_stable_layer_norm:
            self.encoder = FlaxWav2Vec2StableLayerNormEncoder(self.config, dtype=self.dtype)
        else:
            raise NotImplementedError('``config.do_stable_layer_norm is False`` is currently not supported.')
        self.adapter = FlaxWav2Vec2Adapter(self.config, dtype=self.dtype) if self.config.add_adapter else None

    def __call__(self, input_values, attention_mask=None, mask_time_indices=None, deterministic=True, output_attentions=None, output_hidden_states=None, freeze_feature_encoder=False, return_dict=None):
        extract_features = self.feature_extractor(input_values, freeze_feature_encoder=freeze_feature_encoder)
        if attention_mask is not None:
            attention_mask = self._get_feature_vector_attention_mask(extract_features.shape[1], attention_mask, add_adapter=False)
        hidden_states, extract_features = self.feature_projection(extract_features, deterministic=deterministic)
        if mask_time_indices is not None:
            hidden_states = jnp.where(jnp.broadcast_to(mask_time_indices[:, :, None], hidden_states.shape), jnp.broadcast_to(self.masked_spec_embed[None, None, :], hidden_states.shape), hidden_states)
        encoder_outputs = self.encoder(hidden_states, attention_mask=attention_mask, deterministic=deterministic, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        hidden_states = encoder_outputs[0]
        if self.adapter is not None:
            hidden_states = self.adapter(hidden_states)
        if not return_dict:
            return (hidden_states, extract_features) + encoder_outputs[1:]
        return FlaxWav2Vec2BaseModelOutput(last_hidden_state=hidden_states, extract_features=extract_features, hidden_states=encoder_outputs.hidden_states, attentions=encoder_outputs.attentions)

    def _get_feat_extract_output_lengths(self, input_lengths: Union[jnp.ndarray, int], add_adapter: Optional[bool]=None):
        """
        Computes the output length of the convolutional layers
        """
        add_adapter = self.config.add_adapter if add_adapter is None else add_adapter

        def _conv_out_length(input_length, kernel_size, stride):
            return (input_length - kernel_size) // stride + 1
        for kernel_size, stride in zip(self.config.conv_kernel, self.config.conv_stride):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)
        if add_adapter:
            for _ in range(self.config.num_adapter_layers):
                input_lengths = _conv_out_length(input_lengths, 1, self.config.adapter_stride)
        return input_lengths

    def _get_feature_vector_attention_mask(self, feature_vector_length: int, attention_mask: jnp.ndarray, add_adapter=None):
        non_padded_lengths = attention_mask.cumsum(axis=-1)[:, -1]
        output_lengths = self._get_feat_extract_output_lengths(non_padded_lengths, add_adapter=add_adapter)
        batch_size = attention_mask.shape[0]
        attention_mask = jnp.zeros((batch_size, feature_vector_length), dtype=attention_mask.dtype)
        attention_mask = attention_mask.at[jnp.arange(attention_mask.shape[0]), output_lengths - 1].set(1)
        attention_mask = jnp.flip(jnp.flip(attention_mask, -1).cumsum(-1), -1).astype('bool')
        return attention_mask
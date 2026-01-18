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
class FlaxWav2Vec2ForPreTrainingModule(nn.Module):
    config: Wav2Vec2Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.wav2vec2 = FlaxWav2Vec2Module(self.config, dtype=self.dtype)
        self.dropout_features = nn.Dropout(self.config.feat_quantizer_dropout)
        self.quantizer = FlaxWav2Vec2GumbelVectorQuantizer(self.config, dtype=self.dtype)
        self.project_q = nn.Dense(self.config.proj_codevector_dim, kernel_init=jax.nn.initializers.normal(self.config.initializer_range), dtype=self.dtype)
        self.project_hid = nn.Dense(self.config.proj_codevector_dim, kernel_init=jax.nn.initializers.normal(self.config.initializer_range), dtype=self.dtype)

    def __call__(self, input_values, attention_mask=None, mask_time_indices=None, gumbel_temperature: int=1, deterministic: bool=True, output_attentions=None, output_hidden_states=None, freeze_feature_encoder=False, return_dict=None):
        """
        Returns:

        Example:

        ```python

        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.wav2vec2(input_values, attention_mask=attention_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, mask_time_indices=mask_time_indices, deterministic=deterministic, freeze_feature_encoder=freeze_feature_encoder, return_dict=return_dict)
        transformer_features = self.project_hid(outputs[0])
        extract_features = self.dropout_features(outputs[1], deterministic=deterministic)
        quantized_features, codevector_perplexity = self.quantizer(extract_features, mask_time_indices, deterministic=deterministic, temperature=gumbel_temperature)
        quantized_features = self.project_q(quantized_features)
        if not return_dict:
            return (transformer_features, quantized_features, codevector_perplexity) + outputs[2:]
        return FlaxWav2Vec2ForPreTrainingOutput(projected_states=transformer_features, projected_quantized_states=quantized_features, codevector_perplexity=codevector_perplexity, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

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
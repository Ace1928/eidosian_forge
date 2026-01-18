from __future__ import annotations
import collections
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import ACT2FN
from ...modeling_tf_outputs import TFBaseModelOutput
from ...modeling_tf_utils import TFModelInputType, TFPreTrainedModel, keras, shape_list, unpack_inputs
from ...tf_utils import flatten, functional_layernorm
from ...utils import ModelOutput, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_sam import SamConfig, SamMaskDecoderConfig, SamPromptEncoderConfig, SamVisionConfig
class TFSamLayerNorm(keras.layers.Layer):
    """LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with shape (batch_size, height,
    width, channels) while channels_first corresponds to inputs with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-06, data_format='channels_last', **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = normalized_shape
        if self.data_format not in ['channels_last', 'channels_first']:
            raise NotImplementedError(f'Unsupported data format: {self.data_format}')

    def build(self, input_shape):
        self.weight = self.add_weight(shape=self.normalized_shape, initializer='ones', name='weight')
        self.bias = self.add_weight(shape=self.normalized_shape, initializer='zeros', name='bias')
        super().build(input_shape)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        if self.data_format == 'channels_last':
            x = functional_layernorm(x, weight=self.weight, bias=self.bias, epsilon=self.eps, axis=-1)
        elif self.data_format == 'channels_first':
            x = functional_layernorm(x, weight=self.weight, bias=self.bias, epsilon=self.eps, axis=1)
        return x
from __future__ import annotations
import warnings
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import TFBaseModelOutput, TFCausalLMOutput, TFSequenceClassifierOutput
from ...modeling_tf_utils import (
from ...tf_utils import shape_list, stable_softmax
from ...utils import (
from .configuration_wav2vec2 import Wav2Vec2Config
class TFWav2Vec2GroupNorm(keras.layers.Layer):
    """
    From tensorflow-addons https://www.tensorflow.org/addons/api_docs/python/tfa/layers/GroupNormalization
    """

    def __init__(self, groups: int=32, axis: int=-1, epsilon: float=0.001, center: bool=True, scale: bool=True, beta_initializer: keras.initializers.Initializer='zeros', gamma_initializer: keras.initializers.Initializer='ones', beta_regularizer: keras.regularizers.Regularizer=None, gamma_regularizer: keras.regularizers.Regularizer=None, beta_constraint: keras.constraints.Constraint=None, gamma_constraint: keras.constraints.Constraint=None, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.groups = groups
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = keras.initializers.get(beta_initializer)
        self.gamma_initializer = keras.initializers.get(gamma_initializer)
        self.beta_regularizer = keras.regularizers.get(beta_regularizer)
        self.gamma_regularizer = keras.regularizers.get(gamma_regularizer)
        self.beta_constraint = keras.constraints.get(beta_constraint)
        self.gamma_constraint = keras.constraints.get(gamma_constraint)
        self._check_axis()

    def build(self, input_shape):
        self._check_if_input_shape_is_none(input_shape)
        self._set_number_of_groups_for_instance_norm(input_shape)
        self._check_size_of_dimensions(input_shape)
        self._create_input_spec(input_shape)
        self._add_gamma_weight(input_shape)
        self._add_beta_weight(input_shape)
        self.built = True
        super().build(input_shape)

    def call(self, inputs):
        input_shape = keras.backend.int_shape(inputs)
        tensor_input_shape = tf.shape(inputs)
        reshaped_inputs, group_shape = self._reshape_into_groups(inputs, input_shape, tensor_input_shape)
        normalized_inputs = self._apply_normalization(reshaped_inputs, input_shape)
        is_instance_norm = input_shape[self.axis] // self.groups == 1
        if not is_instance_norm:
            outputs = tf.reshape(normalized_inputs, tensor_input_shape)
        else:
            outputs = normalized_inputs
        return outputs

    def get_config(self):
        config = {'groups': self.groups, 'axis': self.axis, 'epsilon': self.epsilon, 'center': self.center, 'scale': self.scale, 'beta_initializer': keras.initializers.serialize(self.beta_initializer), 'gamma_initializer': keras.initializers.serialize(self.gamma_initializer), 'beta_regularizer': keras.regularizers.serialize(self.beta_regularizer), 'gamma_regularizer': keras.regularizers.serialize(self.gamma_regularizer), 'beta_constraint': keras.constraints.serialize(self.beta_constraint), 'gamma_constraint': keras.constraints.serialize(self.gamma_constraint)}
        base_config = super().get_config()
        return {**base_config, **config}

    def compute_output_shape(self, input_shape):
        return input_shape

    def _reshape_into_groups(self, inputs, input_shape, tensor_input_shape):
        group_shape = [tensor_input_shape[i] for i in range(len(input_shape))]
        is_instance_norm = input_shape[self.axis] // self.groups == 1
        if not is_instance_norm:
            group_shape[self.axis] = input_shape[self.axis] // self.groups
            group_shape.insert(self.axis, self.groups)
            group_shape = tf.stack(group_shape)
            reshaped_inputs = tf.reshape(inputs, group_shape)
            return (reshaped_inputs, group_shape)
        else:
            return (inputs, group_shape)

    def _apply_normalization(self, reshaped_inputs, input_shape):
        group_shape = keras.backend.int_shape(reshaped_inputs)
        group_reduction_axes = list(range(1, len(group_shape)))
        is_instance_norm = input_shape[self.axis] // self.groups == 1
        if not is_instance_norm:
            axis = -2 if self.axis == -1 else self.axis - 1
        else:
            axis = -1 if self.axis == -1 else self.axis - 1
        group_reduction_axes.pop(axis)
        mean, variance = tf.nn.moments(reshaped_inputs, group_reduction_axes, keepdims=True)
        gamma, beta = self._get_reshaped_weights(input_shape)
        normalized_inputs = tf.nn.batch_normalization(reshaped_inputs, mean=mean, variance=variance, scale=gamma, offset=beta, variance_epsilon=self.epsilon)
        return normalized_inputs

    def _get_reshaped_weights(self, input_shape):
        broadcast_shape = self._create_broadcast_shape(input_shape)
        gamma = None
        beta = None
        if self.scale:
            gamma = tf.reshape(self.gamma, broadcast_shape)
        if self.center:
            beta = tf.reshape(self.beta, broadcast_shape)
        return (gamma, beta)

    def _check_if_input_shape_is_none(self, input_shape):
        dim = input_shape[self.axis]
        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of input tensor should have a defined dimension but the layer received an input with shape ' + str(input_shape) + '.')

    def _set_number_of_groups_for_instance_norm(self, input_shape):
        dim = input_shape[self.axis]
        if self.groups == -1:
            self.groups = dim

    def _check_size_of_dimensions(self, input_shape):
        dim = input_shape[self.axis]
        if dim < self.groups:
            raise ValueError('Number of groups (' + str(self.groups) + ') cannot be more than the number of channels (' + str(dim) + ').')
        if dim % self.groups != 0:
            raise ValueError('Number of groups (' + str(self.groups) + ') must be a multiple of the number of channels (' + str(dim) + ').')

    def _check_axis(self):
        if self.axis == 0:
            raise ValueError('You are trying to normalize your batch axis. Do you want to use tf.layer.batch_normalization instead')

    def _create_input_spec(self, input_shape):
        dim = input_shape[self.axis]
        self.input_spec = keras.layers.InputSpec(ndim=len(input_shape), axes={self.axis: dim})

    def _add_gamma_weight(self, input_shape):
        dim = input_shape[self.axis]
        shape = (dim,)
        if self.scale:
            self.gamma = self.add_weight(shape=shape, name='gamma', initializer=self.gamma_initializer, regularizer=self.gamma_regularizer, constraint=self.gamma_constraint)
        else:
            self.gamma = None

    def _add_beta_weight(self, input_shape):
        dim = input_shape[self.axis]
        shape = (dim,)
        if self.center:
            self.beta = self.add_weight(shape=shape, name='beta', initializer=self.beta_initializer, regularizer=self.beta_regularizer, constraint=self.beta_constraint)
        else:
            self.beta = None

    def _create_broadcast_shape(self, input_shape):
        broadcast_shape = [1] * len(input_shape)
        is_instance_norm = input_shape[self.axis] // self.groups == 1
        if not is_instance_norm:
            broadcast_shape[self.axis] = input_shape[self.axis] // self.groups
            broadcast_shape.insert(self.axis, self.groups)
        else:
            broadcast_shape[self.axis] = self.groups
        return broadcast_shape
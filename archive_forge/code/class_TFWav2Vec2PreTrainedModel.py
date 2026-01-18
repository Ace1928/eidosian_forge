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
class TFWav2Vec2PreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = Wav2Vec2Config
    base_model_prefix = 'wav2vec2'
    main_input_name = 'input_values'

    @property
    def input_signature(self):
        return {'input_values': tf.TensorSpec((None, None), tf.float32, name='input_values'), 'attention_mask': tf.TensorSpec((None, None), tf.float32, name='attention_mask')}

    @property
    def dummy_inputs(self):
        return {'input_values': tf.random.uniform(shape=(1, 500), dtype=tf.float32), 'attention_mask': tf.ones(shape=(1, 500), dtype=tf.float32)}

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        logger.warning(f'\n{self.__class__.__name__} has backpropagation operations that are NOT supported on CPU. If you wish to train/fine-tune this model, you need a GPU or a TPU')

    def _get_feat_extract_output_lengths(self, input_lengths, add_adapter=None):
        """
        Computes the output length of the convolutional layers
        """
        add_adapter = self.config.add_adapter if add_adapter is None else add_adapter

        def _conv_out_length(input_length, kernel_size, stride):
            return tf.math.floordiv(input_length - kernel_size, stride) + 1
        for kernel_size, stride in zip(self.config.conv_kernel, self.config.conv_stride):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)
        if add_adapter:
            for _ in range(self.config.num_adapter_layers):
                input_lengths = _conv_out_length(input_lengths, 1, self.config.adapter_stride)
        return input_lengths

    def _get_feature_vector_attention_mask(self, feature_vector_length: int, attention_mask: tf.Tensor, add_adapter=None):
        non_padded_lengths = tf.math.cumsum(attention_mask, axis=-1)[:, -1]
        output_lengths = self._get_feat_extract_output_lengths(non_padded_lengths, add_adapter=add_adapter)
        output_lengths = tf.cast(output_lengths, tf.int32)
        batch_size = tf.shape(attention_mask)[0]
        attention_mask = tf.zeros((batch_size, feature_vector_length), dtype=attention_mask.dtype, name='attention_mask')
        attention_mask = tf.tensor_scatter_nd_update(attention_mask, indices=tf.stack([tf.range(batch_size), output_lengths - 1], axis=1), updates=tf.ones([batch_size], dtype=attention_mask.dtype))
        attention_mask = tf.reverse(attention_mask, axis=[-1])
        attention_mask = tf.cumsum(attention_mask, axis=-1)
        attention_mask = tf.reverse(attention_mask, axis=[-1])
        attention_mask = tf.cast(attention_mask, tf.bool)
        return attention_mask
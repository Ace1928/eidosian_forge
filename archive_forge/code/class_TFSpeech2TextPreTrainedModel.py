from __future__ import annotations
import random
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation, glu
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_speech_to_text import Speech2TextConfig
class TFSpeech2TextPreTrainedModel(TFPreTrainedModel):
    config_class = Speech2TextConfig
    base_model_prefix = 'model'
    main_input_name = 'input_features'
    _keys_to_ignore_on_load_unexpected = ['encoder.embed_positions.weights']

    def _get_feat_extract_output_lengths(self, input_lengths: tf.Tensor):
        """
        Computes the output length of the convolutional layers
        """
        for _ in range(self.config.num_conv_layers):
            input_lengths = (input_lengths - 1) // 2 + 1
        return input_lengths

    @property
    def input_signature(self):
        return {'input_features': tf.TensorSpec((None, None, self.config.input_feat_per_channel * self.config.input_channels), tf.float32, name='input_features'), 'attention_mask': tf.TensorSpec((None, None), tf.int32, name='attention_mask'), 'decoder_input_ids': tf.TensorSpec((None, None), tf.int32, name='decoder_input_ids'), 'decoder_attention_mask': tf.TensorSpec((None, None), tf.int32, name='decoder_attention_mask')}
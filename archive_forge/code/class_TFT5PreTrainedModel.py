from __future__ import annotations
import copy
import itertools
import math
import warnings
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from tensorflow.compiler.tf2xla.python.xla import dynamic_slice
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_t5 import T5Config
class TFT5PreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = T5Config
    base_model_prefix = 'transformer'
    _keys_to_ignore_on_load_unexpected = ['decoder\\Wblock[\\W_0]+layer[\\W_1]+EncDecAttention\\Wrelative_attention_bias']

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        if hasattr(self, 'decoder'):
            self.decoder.embed_tokens = self.shared

    def _shift_right(self, input_ids):
        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id
        assert decoder_start_token_id is not None, 'self.model.config.decoder_start_token_id has to be defined. In TF T5 it is usually set to the pad_token_id. See T5 docs for more information'
        start_tokens = tf.fill((shape_list(input_ids)[0], 1), decoder_start_token_id)
        start_tokens = tf.cast(start_tokens, input_ids.dtype)
        shifted_input_ids = tf.concat([start_tokens, input_ids[:, :-1]], -1)
        assert pad_token_id is not None, 'self.model.config.pad_token_id has to be defined.'
        shifted_input_ids = tf.where(shifted_input_ids == -100, tf.cast(tf.fill(shape_list(shifted_input_ids), pad_token_id), shifted_input_ids.dtype), shifted_input_ids)
        assert_gte0 = tf.debugging.assert_greater_equal(shifted_input_ids, tf.constant(0, dtype=shifted_input_ids.dtype))
        with tf.control_dependencies([assert_gte0]):
            shifted_input_ids = tf.identity(shifted_input_ids)
        return shifted_input_ids
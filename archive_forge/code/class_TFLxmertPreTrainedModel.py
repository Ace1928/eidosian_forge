from __future__ import annotations
import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, stable_softmax
from ...utils import (
from .configuration_lxmert import LxmertConfig
class TFLxmertPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = LxmertConfig
    base_model_prefix = 'lxmert'

    @property
    def dummy_inputs(self):
        """
        Dummy inputs to build the network.

        Returns:
            tf.Tensor with dummy inputs
        """
        batch_size = 2
        num_visual_features = 10
        input_ids = tf.constant([[3, 5, 6], [2, 3, 4]], dtype=tf.int32)
        visual_feats = tf.random.uniform((batch_size, num_visual_features, self.config.visual_feat_dim))
        visual_pos = tf.random.uniform((batch_size, num_visual_features, 4))
        return {'input_ids': input_ids, 'visual_feats': visual_feats, 'visual_pos': visual_pos}

    @property
    def input_signature(self):
        return {'input_ids': tf.TensorSpec((None, None), tf.int32, name='input_ids'), 'attention_mask': tf.TensorSpec((None, None), tf.int32, name='attention_mask'), 'visual_feats': tf.TensorSpec((None, None, self.config.visual_feat_dim), tf.float32, name='visual_feats'), 'visual_pos': tf.TensorSpec((None, None, 4), tf.float32, name='visual_pos'), 'visual_attention_mask': tf.TensorSpec((None, None), tf.int32, name='visual_attention_mask'), 'token_type_ids': tf.TensorSpec((None, None), tf.int32, name='token_type_ids')}
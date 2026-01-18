from __future__ import annotations
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_xlnet import XLNetConfig
class TFXLNetLayer(keras.layers.Layer):

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.rel_attn = TFXLNetRelativeAttention(config, name='rel_attn')
        self.ff = TFXLNetFeedForward(config, name='ff')
        self.dropout = keras.layers.Dropout(config.dropout)

    def call(self, output_h, output_g, non_tgt_mask, attn_mask, pos_emb, seg_mat, mems: np.ndarray | tf.Tensor | None=None, target_mapping: np.ndarray | tf.Tensor | None=None, head_mask: np.ndarray | tf.Tensor | None=None, output_attentions: Optional[bool]=False, training: bool=False):
        outputs = self.rel_attn(output_h, output_g, non_tgt_mask, attn_mask, pos_emb, seg_mat, mems, target_mapping, head_mask, output_attentions, training=training)
        output_h, output_g = outputs[:2]
        if output_g is not None:
            output_g = self.ff(output_g, training=training)
        output_h = self.ff(output_h, training=training)
        outputs = (output_h, output_g) + outputs[2:]
        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'rel_attn', None) is not None:
            with tf.name_scope(self.rel_attn.name):
                self.rel_attn.build(None)
        if getattr(self, 'ff', None) is not None:
            with tf.name_scope(self.ff.name):
                self.ff.build(None)
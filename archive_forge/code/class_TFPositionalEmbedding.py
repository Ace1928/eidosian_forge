from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ....modeling_tf_utils import (
from ....tf_utils import shape_list, stable_softmax
from ....utils import (
from .configuration_transfo_xl import TransfoXLConfig
from .modeling_tf_transfo_xl_utilities import TFAdaptiveSoftmaxMask
class TFPositionalEmbedding(keras.layers.Layer):

    def __init__(self, demb, **kwargs):
        super().__init__(**kwargs)
        self.inv_freq = 1 / 10000 ** (tf.range(0, demb, 2.0) / demb)

    def call(self, pos_seq, bsz=None):
        self.inv_freq = tf.cast(self.inv_freq, dtype=pos_seq.dtype)
        sinusoid_inp = tf.einsum('i,j->ij', pos_seq, self.inv_freq)
        pos_emb = tf.concat([tf.sin(sinusoid_inp), tf.cos(sinusoid_inp)], -1)
        if bsz is not None:
            return tf.tile(pos_emb[:, None, :], [1, bsz, 1])
        else:
            return pos_emb[:, None, :]
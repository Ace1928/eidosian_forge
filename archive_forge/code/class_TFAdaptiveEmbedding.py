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
class TFAdaptiveEmbedding(keras.layers.Layer):

    def __init__(self, n_token, d_embed, d_proj, cutoffs, div_val=1, init_std=0.02, sample_softmax=False, **kwargs):
        super().__init__(**kwargs)
        self.n_token = n_token
        self.d_embed = d_embed
        self.init_std = init_std
        self.cutoffs = cutoffs + [n_token]
        self.div_val = div_val
        self.d_proj = d_proj
        self.emb_scale = d_proj ** 0.5
        self.cutoff_ends = [0] + self.cutoffs
        self.emb_layers = []
        self.emb_projs = []
        if div_val == 1:
            raise NotImplementedError
        else:
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = (self.cutoff_ends[i], self.cutoff_ends[i + 1])
                d_emb_i = d_embed // div_val ** i
                self.emb_layers.append(TFTransfoEmbeddings(r_idx - l_idx, d_emb_i, init_std, name=f'emb_layers_._{i}'))

    def build(self, input_shape):
        for i in range(len(self.cutoffs)):
            d_emb_i = self.d_embed // self.div_val ** i
            self.emb_projs.append(self.add_weight(shape=(d_emb_i, self.d_proj), initializer=get_initializer(self.init_std), trainable=True, name=f'emb_projs_._{i}'))
        super().build(input_shape)

    def call(self, inp):
        if self.div_val == 1:
            raise NotImplementedError
        else:
            inp_flat = tf.reshape(inp, (-1,))
            emb_flat = tf.zeros([shape_list(inp_flat)[0], self.d_proj])
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = (self.cutoff_ends[i], self.cutoff_ends[i + 1])
                mask_i = (inp_flat >= l_idx) & (inp_flat < r_idx)
                inp_i = tf.boolean_mask(inp_flat, mask_i) - l_idx
                emb_i = self.emb_layers[i](inp_i)
                emb_i = tf.einsum('id,de->ie', emb_i, self.emb_projs[i])
                mask_idx = tf.where(mask_i)
                scatter = tf.scatter_nd(mask_idx, emb_i, shape_list(emb_flat))
                emb_flat = tf.cast(emb_flat, dtype=scatter.dtype)
                emb_flat += scatter
            embed_shape = shape_list(inp) + [self.d_proj]
            embed = tf.reshape(emb_flat, embed_shape)
        embed *= self.emb_scale
        return embed
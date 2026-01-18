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
class TFLxmertXLayer(keras.layers.Layer):

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.visual_attention = TFLxmertCrossAttentionLayer(config, name='visual_attention')
        self.lang_self_att = TFLxmertSelfAttentionLayer(config, name='lang_self_att')
        self.visn_self_att = TFLxmertSelfAttentionLayer(config, name='visn_self_att')
        self.lang_inter = TFLxmertIntermediate(config, name='lang_inter')
        self.lang_output = TFLxmertOutput(config, name='lang_output')
        self.visn_inter = TFLxmertIntermediate(config, name='visn_inter')
        self.visn_output = TFLxmertOutput(config, name='visn_output')

    def cross_att(self, lang_input, lang_attention_mask, visn_input, visn_attention_mask, output_attentions, training=False):
        lang_attention_lang_input = tf.identity(lang_input)
        visn_attention_lang_input = tf.identity(lang_input)
        lang_attention_visn_input = tf.identity(visn_input)
        visn_attention_visn_input = tf.identity(visn_input)
        lang_att_output = self.visual_attention(lang_attention_lang_input, lang_attention_visn_input, visn_attention_mask, output_attentions=output_attentions, training=training)
        visn_att_output = self.visual_attention(visn_attention_visn_input, visn_attention_lang_input, lang_attention_mask, output_attentions=output_attentions, training=training)
        return (lang_att_output, visn_att_output)

    def self_att(self, lang_input, lang_attention_mask, visn_input, visn_attention_mask, training=False):
        output_attentions = False
        lang_att_output = self.lang_self_att(lang_input, lang_attention_mask, output_attentions, training=training)
        visn_att_output = self.visn_self_att(visn_input, visn_attention_mask, output_attentions, training=training)
        return (lang_att_output[0], visn_att_output[0])

    def output_fc(self, lang_input, visn_input, training=False):
        lang_inter_output = self.lang_inter(lang_input)
        visn_inter_output = self.visn_inter(visn_input)
        lang_output = self.lang_output(lang_inter_output, lang_input, training)
        visn_output = self.visn_output(visn_inter_output, visn_input, training)
        return (lang_output, visn_output)

    def call(self, lang_feats, lang_attention_mask, visn_feats, visn_attention_mask, output_attentions, training=False):
        lang_att_output = lang_feats
        visn_att_output = visn_feats
        lang_att_output, visn_att_output = self.cross_att(lang_att_output, lang_attention_mask, visn_att_output, visn_attention_mask, output_attentions, training=training)
        attention_probs = lang_att_output[1:]
        lang_att_output, visn_att_output = self.self_att(lang_att_output[0], lang_attention_mask, visn_att_output[0], visn_attention_mask, training=training)
        lang_output, visn_output = self.output_fc(lang_att_output, visn_att_output, training=training)
        return (lang_output, visn_output, attention_probs[0]) if output_attentions else (lang_output, visn_output)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'visual_attention', None) is not None:
            with tf.name_scope(self.visual_attention.name):
                self.visual_attention.build(None)
        if getattr(self, 'lang_self_att', None) is not None:
            with tf.name_scope(self.lang_self_att.name):
                self.lang_self_att.build(None)
        if getattr(self, 'visn_self_att', None) is not None:
            with tf.name_scope(self.visn_self_att.name):
                self.visn_self_att.build(None)
        if getattr(self, 'lang_inter', None) is not None:
            with tf.name_scope(self.lang_inter.name):
                self.lang_inter.build(None)
        if getattr(self, 'lang_output', None) is not None:
            with tf.name_scope(self.lang_output.name):
                self.lang_output.build(None)
        if getattr(self, 'visn_inter', None) is not None:
            with tf.name_scope(self.visn_inter.name):
                self.visn_inter.build(None)
        if getattr(self, 'visn_output', None) is not None:
            with tf.name_scope(self.visn_output.name):
                self.visn_output.build(None)
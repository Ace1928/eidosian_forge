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
class TFSamTwoWayTransformer(keras.layers.Layer):

    def __init__(self, config: SamMaskDecoderConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.num_hidden_layers = config.num_hidden_layers
        self.layers = []
        for i in range(self.num_hidden_layers):
            self.layers.append(TFSamTwoWayAttentionBlock(config, skip_first_layer_pe=i == 0, name=f'layers_._{i}'))
        self.final_attn_token_to_image = TFSamAttention(config, name='final_attn_token_to_image')
        self.layer_norm_final_attn = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='layer_norm_final_attn')

    def call(self, point_embeddings: tf.Tensor, image_embeddings: tf.Tensor, image_positional_embeddings: tf.Tensor, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, TFBaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        all_attentions = ()
        if image_embeddings is None:
            raise ValueError('You have to specify an image_embedding')
        image_embeddings = tf.transpose(flatten(image_embeddings, 2), perm=(0, 2, 1))[:, None]
        image_positional_embeddings = tf.transpose(flatten(image_positional_embeddings, 2), (0, 2, 1))[:, None]
        queries = point_embeddings
        keys = image_embeddings
        for layer in self.layers:
            queries, keys, attention_outputs = layer(queries=queries, keys=keys, query_point_embedding=point_embeddings, key_point_embedding=image_positional_embeddings, output_attentions=output_attentions)
            if output_attentions:
                all_attentions = all_attentions + (attention_outputs,)
        query = queries + point_embeddings
        key = keys + image_positional_embeddings
        attn_out = self.final_attn_token_to_image(query=query, key=key, value=keys)
        queries = queries + attn_out
        queries = self.layer_norm_final_attn(queries)
        return (queries, keys, all_attentions)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'final_attn_token_to_image', None) is not None:
            with tf.name_scope(self.final_attn_token_to_image.name):
                self.final_attn_token_to_image.build(None)
        if getattr(self, 'layer_norm_final_attn', None) is not None:
            with tf.name_scope(self.layer_norm_final_attn.name):
                self.layer_norm_final_attn.build([None, None, None, self.config.hidden_size])
        for layer in self.layers:
            with tf.name_scope(layer.name):
                layer.build(None)
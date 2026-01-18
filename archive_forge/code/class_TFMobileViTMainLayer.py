from __future__ import annotations
from typing import Dict, Optional, Tuple, Union
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...file_utils import (
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import shape_list, stable_softmax
from ...utils import logging
from .configuration_mobilevit import MobileViTConfig
@keras_serializable
class TFMobileViTMainLayer(keras.layers.Layer):
    config_class = MobileViTConfig

    def __init__(self, config: MobileViTConfig, expand_output: bool=True, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.expand_output = expand_output
        self.conv_stem = TFMobileViTConvLayer(config, in_channels=config.num_channels, out_channels=config.neck_hidden_sizes[0], kernel_size=3, stride=2, name='conv_stem')
        self.encoder = TFMobileViTEncoder(config, name='encoder')
        if self.expand_output:
            self.conv_1x1_exp = TFMobileViTConvLayer(config, in_channels=config.neck_hidden_sizes[5], out_channels=config.neck_hidden_sizes[6], kernel_size=1, name='conv_1x1_exp')
        self.pooler = keras.layers.GlobalAveragePooling2D(data_format='channels_first', name='pooler')

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        raise NotImplementedError

    @unpack_inputs
    def call(self, pixel_values: tf.Tensor | None=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: bool=False) -> Union[Tuple[tf.Tensor], TFBaseModelOutputWithPooling]:
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        pixel_values = tf.transpose(pixel_values, perm=(0, 2, 3, 1))
        embedding_output = self.conv_stem(pixel_values, training=training)
        encoder_outputs = self.encoder(embedding_output, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        if self.expand_output:
            last_hidden_state = self.conv_1x1_exp(encoder_outputs[0])
            last_hidden_state = tf.transpose(last_hidden_state, perm=[0, 3, 1, 2])
            pooled_output = self.pooler(last_hidden_state)
        else:
            last_hidden_state = encoder_outputs[0]
            last_hidden_state = tf.transpose(last_hidden_state, perm=[0, 3, 1, 2])
            pooled_output = None
        if not return_dict:
            output = (last_hidden_state, pooled_output) if pooled_output is not None else (last_hidden_state,)
            if not self.expand_output:
                remaining_encoder_outputs = encoder_outputs[1:]
                remaining_encoder_outputs = tuple([tf.transpose(h, perm=(0, 3, 1, 2)) for h in remaining_encoder_outputs[0]])
                remaining_encoder_outputs = (remaining_encoder_outputs,)
                return output + remaining_encoder_outputs
            else:
                return output + encoder_outputs[1:]
        if output_hidden_states:
            hidden_states = tuple([tf.transpose(h, perm=(0, 3, 1, 2)) for h in encoder_outputs[1]])
        return TFBaseModelOutputWithPooling(last_hidden_state=last_hidden_state, pooler_output=pooled_output, hidden_states=hidden_states if output_hidden_states else encoder_outputs.hidden_states)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'conv_stem', None) is not None:
            with tf.name_scope(self.conv_stem.name):
                self.conv_stem.build(None)
        if getattr(self, 'encoder', None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)
        if getattr(self, 'pooler', None) is not None:
            with tf.name_scope(self.pooler.name):
                self.pooler.build([None, None, None, None])
        if getattr(self, 'conv_1x1_exp', None) is not None:
            with tf.name_scope(self.conv_1x1_exp.name):
                self.conv_1x1_exp.build(None)
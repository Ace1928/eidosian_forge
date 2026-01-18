from __future__ import annotations
import collections.abc
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import tensorflow as tf
from ...modeling_tf_outputs import TFImageClassifierOutputWithNoAttention
from ...modeling_tf_utils import (
from ...tf_utils import shape_list, stable_softmax
from ...utils import (
from .configuration_cvt import CvtConfig
class TFCvtEncoder(keras.layers.Layer):
    """
    Convolutional Vision Transformer encoder. CVT has 3 stages of encoder blocks with their respective number of layers
    (depth) being 1, 2 and 10.

    Args:
        config ([`CvtConfig`]): Model configuration class.
    """
    config_class = CvtConfig

    def __init__(self, config: CvtConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.stages = [TFCvtStage(config, stage_idx, name=f'stages.{stage_idx}') for stage_idx in range(len(config.depth))]

    def call(self, pixel_values: TFModelInputType, output_hidden_states: Optional[bool]=False, return_dict: Optional[bool]=True, training: Optional[bool]=False) -> Union[TFBaseModelOutputWithCLSToken, Tuple[tf.Tensor]]:
        all_hidden_states = () if output_hidden_states else None
        hidden_state = pixel_values
        hidden_state = tf.transpose(hidden_state, perm=(0, 2, 3, 1))
        cls_token = None
        for _, stage_module in enumerate(self.stages):
            hidden_state, cls_token = stage_module(hidden_state, training=training)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_state,)
        hidden_state = tf.transpose(hidden_state, perm=(0, 3, 1, 2))
        if output_hidden_states:
            all_hidden_states = tuple([tf.transpose(hs, perm=(0, 3, 1, 2)) for hs in all_hidden_states])
        if not return_dict:
            return tuple((v for v in [hidden_state, cls_token, all_hidden_states] if v is not None))
        return TFBaseModelOutputWithCLSToken(last_hidden_state=hidden_state, cls_token_value=cls_token, hidden_states=all_hidden_states)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'stages', None) is not None:
            for layer in self.stages:
                with tf.name_scope(layer.name):
                    layer.build(None)
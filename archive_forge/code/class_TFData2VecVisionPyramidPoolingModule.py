from __future__ import annotations
import collections.abc
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import shape_list, stable_softmax
from ...utils import (
from .configuration_data2vec_vision import Data2VecVisionConfig
class TFData2VecVisionPyramidPoolingModule(keras.layers.Layer):
    """
    Pyramid Pooling Module (PPM) used in PSPNet.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module.
        channels (int): Channels after modules, before conv_seg.

    Based on OpenMMLab's implementation, found in https://github.com/open-mmlab/mmsegmentation.
    """

    def __init__(self, pool_scales: Tuple[int, ...], in_channels: int, out_channels: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.pool_scales = pool_scales
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layer_list = []
        for idx, pool_scale in enumerate(pool_scales):
            pool_scale = pool_scale if isinstance(pool_scale, collections.abc.Iterable) else (pool_scale, pool_scale)
            self.layer_list.append([TFAdaptiveAvgPool2D(output_dims=pool_scale), TFData2VecVisionConvModule(in_channels=in_channels, out_channels=self.out_channels, kernel_size=1, name=f'{idx}.1')])

    def call(self, x: tf.Tensor) -> List[tf.Tensor]:
        ppm_outs = []
        inputs = x
        for ppm in self.layer_list:
            for layer_module in ppm:
                ppm_out = layer_module(x)
                x = ppm_out
            upsampled_ppm_out = tf.image.resize(ppm_out, size=shape_list(inputs)[1:-1], method='bilinear')
            ppm_outs.append(upsampled_ppm_out)
        return ppm_outs

    def build(self, input_shape=None):
        for layer in self.layer_list:
            for layer_module in layer:
                with tf.name_scope(layer_module.name):
                    layer_module.build(None)
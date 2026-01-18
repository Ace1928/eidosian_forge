from __future__ import annotations
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_mobilebert import MobileBertConfig
class TFLayerNorm(keras.layers.LayerNormalization):

    def __init__(self, feat_size, *args, **kwargs):
        self.feat_size = feat_size
        super().__init__(*args, **kwargs)

    def build(self, input_shape=None):
        super().build([None, None, self.feat_size])
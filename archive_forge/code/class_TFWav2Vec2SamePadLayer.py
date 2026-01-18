from __future__ import annotations
import warnings
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import TFBaseModelOutput, TFCausalLMOutput, TFSequenceClassifierOutput
from ...modeling_tf_utils import (
from ...tf_utils import shape_list, stable_softmax
from ...utils import (
from .configuration_wav2vec2 import Wav2Vec2Config
class TFWav2Vec2SamePadLayer(keras.layers.Layer):

    def __init__(self, num_conv_pos_embeddings, **kwargs):
        super().__init__(**kwargs)
        self.num_pad_remove = 1 if num_conv_pos_embeddings % 2 == 0 else 0

    def call(self, hidden_states):
        if self.num_pad_remove > 0:
            hidden_states = hidden_states[:, :-self.num_pad_remove, :]
        return hidden_states
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
def _reshape_into_groups(self, inputs, input_shape, tensor_input_shape):
    group_shape = [tensor_input_shape[i] for i in range(len(input_shape))]
    is_instance_norm = input_shape[self.axis] // self.groups == 1
    if not is_instance_norm:
        group_shape[self.axis] = input_shape[self.axis] // self.groups
        group_shape.insert(self.axis, self.groups)
        group_shape = tf.stack(group_shape)
        reshaped_inputs = tf.reshape(inputs, group_shape)
        return (reshaped_inputs, group_shape)
    else:
        return (inputs, group_shape)
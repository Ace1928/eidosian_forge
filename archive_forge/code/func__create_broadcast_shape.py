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
def _create_broadcast_shape(self, input_shape):
    broadcast_shape = [1] * len(input_shape)
    is_instance_norm = input_shape[self.axis] // self.groups == 1
    if not is_instance_norm:
        broadcast_shape[self.axis] = input_shape[self.axis] // self.groups
        broadcast_shape.insert(self.axis, self.groups)
    else:
        broadcast_shape[self.axis] = self.groups
    return broadcast_shape
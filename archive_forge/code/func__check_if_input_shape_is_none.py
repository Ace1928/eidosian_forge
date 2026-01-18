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
def _check_if_input_shape_is_none(self, input_shape):
    dim = input_shape[self.axis]
    if dim is None:
        raise ValueError('Axis ' + str(self.axis) + ' of input tensor should have a defined dimension but the layer received an input with shape ' + str(input_shape) + '.')
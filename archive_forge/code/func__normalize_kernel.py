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
def _normalize_kernel(self):
    """Generate normalized weights."""
    kernel = tf.nn.l2_normalize(self.weight_v, axis=self.kernel_norm_axes) * tf.transpose(self.weight_g)
    self.kernel = tf.transpose(kernel)
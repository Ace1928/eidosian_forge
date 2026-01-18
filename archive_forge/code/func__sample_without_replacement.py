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
def _sample_without_replacement(distribution, num_samples):
    """
    Categorical sampling without replacement is currently not implemented. The gumbel-max trick will do for now - see
    https://github.com/tensorflow/tensorflow/issues/9260 for more info
    """
    z = -tf.math.log(tf.random.uniform(shape_list(distribution), 0, 1))
    _, indices = tf.nn.top_k(distribution + z, num_samples)
    return indices
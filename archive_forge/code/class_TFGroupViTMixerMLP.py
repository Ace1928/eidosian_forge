from __future__ import annotations
import collections.abc
import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import TFBaseModelOutput, TFBaseModelOutputWithPooling
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_groupvit import GroupViTConfig, GroupViTTextConfig, GroupViTVisionConfig
class TFGroupViTMixerMLP(TFGroupViTMLP):

    def call(self, x, training: bool=False):
        x = super().call(hidden_states=tf.transpose(x, perm=(0, 2, 1)))
        return tf.transpose(x, perm=(0, 2, 1))
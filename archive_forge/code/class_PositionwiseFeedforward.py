import gymnasium as gym
from gymnasium.spaces import Box, Discrete, MultiDiscrete
import numpy as np
import tree  # pip install dm_tree
from typing import Any, Dict, Optional, Union
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.layers import (
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.recurrent_net import RecurrentNetwork
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.spaces.space_utils import get_base_struct_from_space
from ray.rllib.utils.tf_utils import flatten_inputs_to_1d_tensor, one_hot
from ray.rllib.utils.typing import ModelConfigDict, TensorType, List
from ray.rllib.utils.deprecation import deprecation_warning
from ray.util import log_once
class PositionwiseFeedforward(tf.keras.layers.Layer if tf else object):
    """A 2x linear layer with ReLU activation in between described in [1].

    Each timestep coming from the attention head will be passed through this
    layer separately.
    """

    def __init__(self, out_dim: int, hidden_dim: int, output_activation: Optional[Any]=None, **kwargs):
        super().__init__(**kwargs)
        self._hidden_layer = tf.keras.layers.Dense(hidden_dim, activation=tf.nn.relu)
        self._output_layer = tf.keras.layers.Dense(out_dim, activation=output_activation)
        if log_once('positionwise_feedforward_tf'):
            deprecation_warning(old='rllib.models.tf.attention_net.PositionwiseFeedforward')

    def call(self, inputs: TensorType, **kwargs) -> TensorType:
        del kwargs
        output = self._hidden_layer(inputs)
        return self._output_layer(output)
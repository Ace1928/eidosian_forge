from gymnasium.spaces import Box, Discrete
import numpy as np
from typing import Optional, TYPE_CHECKING, Union
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.tf_action_dist import Categorical, Deterministic
from ray.rllib.models.torch.torch_action_dist import (
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override, PublicAPI
from ray.rllib.utils.exploration.exploration import Exploration
from ray.rllib.utils.framework import get_variable, try_import_tf, try_import_torch
from ray.rllib.utils.from_config import from_config
from ray.rllib.utils.numpy import softmax, SMALL_NUMBER
from ray.rllib.utils.typing import TensorType
def _tf_remove_noise_op(self):
    """Generates a tf-op for removing noise from the model's weights.

        Also used by tf-eager.

        Returns:
            tf.op: The tf op to remve the currently stored noise from the NN.
        """
    remove_noise_ops = list()
    for var, noise in zip(self.model_variables, self.noise):
        remove_noise_ops.append(tf1.assign_add(var, -noise))
    ret = tf.group(*tuple(remove_noise_ops))
    with tf1.control_dependencies([ret]):
        return tf.no_op()
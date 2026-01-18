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
def _sample_new_noise_and_add(self, *, tf_sess=None, override=False):
    if self.framework == 'tf':
        if override and self.weights_are_currently_noisy:
            tf_sess.run(self.tf_remove_noise_op)
        tf_sess.run(self.tf_sample_new_noise_and_add_op)
    else:
        if override and self.weights_are_currently_noisy:
            self._remove_noise()
        self._sample_new_noise()
        self._add_stored_noise()
    self.weights_are_currently_noisy = True
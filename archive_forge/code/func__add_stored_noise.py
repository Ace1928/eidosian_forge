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
def _add_stored_noise(self, *, tf_sess=None):
    """Adds the stored `self.noise` to the model's parameters.

        Note: No new sampling of noise here.

        Args:
            tf_sess (Optional[tf.Session]): The tf-session to use to add the
                stored noise to the (currently noise-free) weights.
            override: If True, undo any currently applied noise first,
                then add the currently stored noise.
        """
    assert self.weights_are_currently_noisy is False
    if self.framework == 'tf':
        tf_sess.run(self.tf_add_stored_noise_op)
    elif self.framework == 'tf2':
        self._tf_add_stored_noise_op()
    else:
        for var, noise in zip(self.model_variables, self.noise):
            var.requires_grad = False
            var.add_(noise)
            var.requires_grad = True
    self.weights_are_currently_noisy = True
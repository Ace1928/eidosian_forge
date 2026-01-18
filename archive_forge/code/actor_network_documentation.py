from typing import Optional
import gymnasium as gym
from gymnasium.spaces import Box, Discrete
import numpy as np
from ray.rllib.algorithms.dreamerv3.tf.models.components.mlp import MLP
from ray.rllib.algorithms.dreamerv3.utils import (
from ray.rllib.utils.framework import try_import_tf, try_import_tfp
Helper method to create an action distribution object from (T, B, ..) params.

        Args:
            action_dist_params_T_B: The time-major action distribution parameters.
                This could be simply the logits (discrete) or a to-be-split-in-2
                tensor for mean and stddev (continuous).

        Returns:
            The tfp action distribution object, from which one can sample, compute
            log probs, entropy, etc..
        
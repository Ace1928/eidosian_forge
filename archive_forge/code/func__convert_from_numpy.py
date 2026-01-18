import gymnasium as gym
import numpy as np
import tree
from collections import defaultdict
from functools import partial
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
from ray.rllib.core.models.base import STATE_IN, STATE_OUT
from ray.rllib.core.rl_module.rl_module import RLModule, SingleAgentRLModuleSpec
from ray.rllib.env.env_runner import EnvRunner
from ray.rllib.env.single_agent_episode import SingleAgentEpisode
from ray.rllib.env.utils import _gym_env_creator
from ray.rllib.evaluation.metrics import RolloutMetrics
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID, SampleBatch
from ray.rllib.utils.annotations import ExperimentalAPI, override
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.torch_utils import convert_to_torch_tensor
from ray.rllib.utils.typing import TensorStructType, TensorType
from ray.tune.registry import ENV_CREATOR, _global_registry
def _convert_from_numpy(self, array: np.array) -> TensorType:
    """Converts a numpy array to a framework-specific tensor."""
    if self.config.framework_str == 'torch':
        return torch.from_numpy(array)
    else:
        return tf.convert_to_tensor(array)
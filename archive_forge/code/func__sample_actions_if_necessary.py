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
def _sample_actions_if_necessary(self, fwd_out: TensorStructType, explore: bool=True) -> Tuple[np.array, np.array]:
    """Samples actions from action distribution if necessary."""
    if SampleBatch.ACTIONS in fwd_out.keys():
        actions = convert_to_numpy(fwd_out[SampleBatch.ACTIONS])
        action_logp = convert_to_numpy(fwd_out[SampleBatch.ACTION_LOGP])
    else:
        if explore:
            action_dist_cls = self.module.get_exploration_action_dist_cls()
        else:
            action_dist_cls = self.module.get_inference_action_dist_cls()
        action_dist = action_dist_cls.from_logits(fwd_out[SampleBatch.ACTION_DIST_INPUTS])
        actions = action_dist.sample()
        action_logp = convert_to_numpy(action_dist.logp(actions))
        actions = convert_to_numpy(actions)
    return (actions, action_logp)
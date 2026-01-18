import collections
from gymnasium.spaces import Space
import logging
import numpy as np
import tree  # pip install dm_tree
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING, Union
from ray.rllib.env.base_env import _DUMMY_AGENT_ID
from ray.rllib.evaluation.collectors.sample_collector import SampleCollector
from ray.rllib.evaluation.collectors.agent_collector import AgentCollector
from ray.rllib.evaluation.episode import Episode
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.policy_map import PolicyMap
from ray.rllib.policy.sample_batch import SampleBatch, MultiAgentBatch, concat_samples
from ray.rllib.utils.annotations import override, PublicAPI
from ray.rllib.utils.debug import summarize
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.spaces.space_utils import get_dummy_batch_for_space
from ray.rllib.utils.typing import (
from ray.util.debug import log_once
def _add_to_next_inference_call(self, agent_key: Tuple[EpisodeID, AgentID]) -> None:
    """Adds an Agent key (episode+agent IDs) to the next inference call.

        This makes sure that the agent's current data (in the trajectory) is
        used for generating the next input_dict for a
        `Policy.compute_actions()` call.

        Args:
            agent_key (Tuple[EpisodeID, AgentID]: A unique agent key (across
                vectorized environments).
        """
    pid = self.agent_key_to_policy_id[agent_key]
    if pid not in self.forward_pass_size:
        assert pid in self.policy_map
        self.forward_pass_size[pid] = 0
        self.forward_pass_agent_keys[pid] = []
    idx = self.forward_pass_size[pid]
    assert idx >= 0
    if idx == 0:
        self.forward_pass_agent_keys[pid].clear()
    self.forward_pass_agent_keys[pid].append(agent_key)
    self.forward_pass_size[pid] += 1
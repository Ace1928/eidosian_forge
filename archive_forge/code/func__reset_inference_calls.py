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
def _reset_inference_calls(self, policy_id: PolicyID) -> None:
    """Resets internal inference input-dict registries.

        Calling `self.get_inference_input_dict()` after this method is called
        would return an empty input-dict.

        Args:
            policy_id: The policy ID for which to reset the
                inference pointers.
        """
    self.forward_pass_size[policy_id] = 0
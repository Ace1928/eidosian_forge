import gymnasium as gym
import logging
from typing import Callable, Dict, List, Tuple, Optional, Union, Set, Type
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.env.env_context import EnvContext
from ray.rllib.utils.annotations import (
from ray.rllib.utils.typing import (
from ray.util import log_once
def _init_env_state(self, idx: Optional[int]=None) -> None:
    """Resets all or one particular sub-environment's state (by index).

        Args:
            idx: The index to reset at. If None, reset all the sub-environments' states.
        """
    if idx is None:
        self.env_states = [_MultiAgentEnvState(env, self.restart_failed_sub_environments) for env in self.envs]
    else:
        assert isinstance(idx, int)
        self.env_states[idx] = _MultiAgentEnvState(self.envs[idx], self.restart_failed_sub_environments)
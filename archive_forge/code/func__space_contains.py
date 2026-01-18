import logging
from typing import Callable, Tuple, Optional, List, Dict, Any, TYPE_CHECKING, Union, Set
import gymnasium as gym
import ray
from ray.rllib.utils.annotations import Deprecated, DeveloperAPI, PublicAPI
from ray.rllib.utils.typing import AgentID, EnvID, EnvType, MultiEnvDict
def _space_contains(self, space: gym.Space, x: MultiEnvDict) -> bool:
    """Check if the given space contains the observations of x.

        Args:
            space: The space to if x's observations are contained in.
            x: The observations to check.

        Returns:
            True if the observations of x are contained in space.
        """
    agents = set(self.get_agent_ids())
    for multi_agent_dict in x.values():
        for agent_id, obs in multi_agent_dict.items():
            if agent_id == _DUMMY_AGENT_ID:
                if not space.contains(obs):
                    return False
            elif agent_id not in agents or not space[agent_id].contains(obs):
                return False
    return True
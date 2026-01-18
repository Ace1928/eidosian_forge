import gymnasium as gym
import logging
from typing import Callable, Dict, List, Optional, Set, Tuple, TYPE_CHECKING
import ray
from ray.util import log_once
from ray.rllib.env.base_env import BaseEnv, _DUMMY_AGENT_ID, ASYNC_RESET_RETURN
from ray.rllib.utils.annotations import override, PublicAPI
from ray.rllib.utils.typing import AgentID, EnvID, EnvType, MultiEnvDict
@ray.remote(num_cpus=0)
class _RemoteMultiAgentEnv:
    """Wrapper class for making a multi-agent env a remote actor."""

    def __init__(self, make_env, i):
        self.env = make_env(i)
        self.agent_ids = set()

    def reset(self, *, seed: Optional[int]=None, options: Optional[dict]=None):
        obs, info = self.env.reset(seed=seed, options=options)
        rew = {}
        for agent_id in obs.keys():
            self.agent_ids.add(agent_id)
            rew[agent_id] = 0.0
        terminated = {'__all__': False}
        truncated = {'__all__': False}
        return (obs, rew, terminated, truncated, info)

    def step(self, action_dict):
        return self.env.step(action_dict)

    def observation_space(self):
        return self.env.observation_space

    def action_space(self):
        return self.env.action_space

    def get_agent_ids(self) -> Set[AgentID]:
        return self.agent_ids
import gymnasium as gym
import logging
from typing import Callable, Dict, List, Optional, Set, Tuple, TYPE_CHECKING
import ray
from ray.util import log_once
from ray.rllib.env.base_env import BaseEnv, _DUMMY_AGENT_ID, ASYNC_RESET_RETURN
from ray.rllib.utils.annotations import override, PublicAPI
from ray.rllib.utils.typing import AgentID, EnvID, EnvType, MultiEnvDict
@ray.remote(num_cpus=0)
class _RemoteSingleAgentEnv:
    """Wrapper class for making a gym env a remote actor."""

    def __init__(self, make_env, i):
        self.env = make_env(i)

    def reset(self, *, seed: Optional[int]=None, options: Optional[dict]=None):
        obs_and_info = self.env.reset(seed=seed, options=options)
        obs = {_DUMMY_AGENT_ID: obs_and_info[0]}
        info = {_DUMMY_AGENT_ID: obs_and_info[1]}
        rew = {_DUMMY_AGENT_ID: 0.0}
        terminated = {'__all__': False}
        truncated = {'__all__': False}
        return (obs, rew, terminated, truncated, info)

    def step(self, action):
        results = self.env.step(action[_DUMMY_AGENT_ID])
        obs, rew, terminated, truncated, info = [{_DUMMY_AGENT_ID: x} for x in results]
        terminated['__all__'] = terminated[_DUMMY_AGENT_ID]
        truncated['__all__'] = truncated[_DUMMY_AGENT_ID]
        return (obs, rew, terminated, truncated, info)

    def observation_space(self):
        return self.env.observation_space

    def action_space(self):
        return self.env.action_space
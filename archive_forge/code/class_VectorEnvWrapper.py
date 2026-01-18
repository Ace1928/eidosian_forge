import logging
import gymnasium as gym
import numpy as np
from typing import Callable, List, Optional, Tuple, Union, Set
from ray.rllib.env.base_env import BaseEnv, _DUMMY_AGENT_ID
from ray.rllib.utils.annotations import Deprecated, override, PublicAPI
from ray.rllib.utils.typing import (
from ray.util import log_once
@PublicAPI
class VectorEnvWrapper(BaseEnv):
    """Internal adapter of VectorEnv to BaseEnv.

    We assume the caller will always send the full vector of actions in each
    call to send_actions(), and that they call reset_at() on all completed
    environments before calling send_actions().
    """

    def __init__(self, vector_env: VectorEnv):
        self.vector_env = vector_env
        self.num_envs = vector_env.num_envs
        self._observation_space = vector_env.observation_space
        self._action_space = vector_env.action_space
        self.new_obs = None
        self.cur_rewards = None
        self.cur_terminateds = None
        self.cur_truncateds = None
        self.cur_infos = None
        self.first_reset_done = False
        self._init_env_state(idx=None)

    @override(BaseEnv)
    def poll(self) -> Tuple[MultiEnvDict, MultiEnvDict, MultiEnvDict, MultiEnvDict, MultiEnvDict, MultiEnvDict]:
        from ray.rllib.env.base_env import with_dummy_agent_id
        if not self.first_reset_done:
            self.first_reset_done = True
            self.new_obs, self.cur_infos = self.vector_env.vector_reset()
        new_obs = dict(enumerate(self.new_obs))
        rewards = dict(enumerate(self.cur_rewards))
        terminateds = dict(enumerate(self.cur_terminateds))
        truncateds = dict(enumerate(self.cur_truncateds))
        infos = dict(enumerate(self.cur_infos))
        self.new_obs = []
        self.cur_rewards = []
        self.cur_terminateds = []
        self.cur_truncateds = []
        self.cur_infos = []
        return (with_dummy_agent_id(new_obs), with_dummy_agent_id(rewards), with_dummy_agent_id(terminateds, '__all__'), with_dummy_agent_id(truncateds, '__all__'), with_dummy_agent_id(infos), {})

    @override(BaseEnv)
    def send_actions(self, action_dict: MultiEnvDict) -> None:
        from ray.rllib.env.base_env import _DUMMY_AGENT_ID
        action_vector = [None] * self.num_envs
        for i in range(self.num_envs):
            action_vector[i] = action_dict[i][_DUMMY_AGENT_ID]
        self.new_obs, self.cur_rewards, self.cur_terminateds, self.cur_truncateds, self.cur_infos = self.vector_env.vector_step(action_vector)

    @override(BaseEnv)
    def try_reset(self, env_id: Optional[EnvID]=None, *, seed: Optional[int]=None, options: Optional[dict]=None) -> Tuple[MultiEnvDict, MultiEnvDict]:
        from ray.rllib.env.base_env import _DUMMY_AGENT_ID
        if env_id is None:
            env_id = 0
        assert isinstance(env_id, int)
        obs, infos = self.vector_env.reset_at(env_id, seed=seed, options=options)
        if isinstance(obs, Exception):
            return ({env_id: obs}, {env_id: infos})
        else:
            return ({env_id: {_DUMMY_AGENT_ID: obs}}, {env_id: {_DUMMY_AGENT_ID: infos}})

    @override(BaseEnv)
    def try_restart(self, env_id: Optional[EnvID]=None) -> None:
        assert env_id is None or isinstance(env_id, int)
        self.vector_env.restart_at(env_id)
        self._init_env_state(env_id)

    @override(BaseEnv)
    def get_sub_environments(self, as_dict: bool=False) -> Union[List[EnvType], dict]:
        if not as_dict:
            return self.vector_env.get_sub_environments()
        else:
            return {_id: env for _id, env in enumerate(self.vector_env.get_sub_environments())}

    @override(BaseEnv)
    def try_render(self, env_id: Optional[EnvID]=None) -> None:
        assert env_id is None or isinstance(env_id, int)
        return self.vector_env.try_render_at(env_id)

    @property
    @override(BaseEnv)
    @PublicAPI
    def observation_space(self) -> gym.Space:
        return self._observation_space

    @property
    @override(BaseEnv)
    @PublicAPI
    def action_space(self) -> gym.Space:
        return self._action_space

    @override(BaseEnv)
    @PublicAPI
    def action_space_sample(self, agent_id: list=None) -> MultiEnvDict:
        del agent_id
        return {0: {_DUMMY_AGENT_ID: self._action_space.sample()}}

    @override(BaseEnv)
    @PublicAPI
    def observation_space_sample(self, agent_id: list=None) -> MultiEnvDict:
        del agent_id
        return {0: {_DUMMY_AGENT_ID: self._observation_space.sample()}}

    @override(BaseEnv)
    @PublicAPI
    def get_agent_ids(self) -> Set[AgentID]:
        return {_DUMMY_AGENT_ID}

    def _init_env_state(self, idx: Optional[int]=None) -> None:
        """Resets all or one particular sub-environment's state (by index).

        Args:
            idx: The index to reset at. If None, reset all the sub-environments' states.
        """
        if idx is None:
            self.new_obs = [None for _ in range(self.num_envs)]
            self.cur_rewards = [0.0 for _ in range(self.num_envs)]
            self.cur_terminateds = [False for _ in range(self.num_envs)]
            self.cur_truncateds = [False for _ in range(self.num_envs)]
            self.cur_infos = [{} for _ in range(self.num_envs)]
        else:
            self.new_obs[idx], self.cur_infos[idx] = self.vector_env.reset_at(idx)
            self.cur_rewards[idx] = 0.0
            self.cur_terminateds[idx] = False
            self.cur_truncateds[idx] = False
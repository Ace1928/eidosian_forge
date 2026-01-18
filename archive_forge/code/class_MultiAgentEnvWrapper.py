import gymnasium as gym
import logging
from typing import Callable, Dict, List, Tuple, Optional, Union, Set, Type
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.env.env_context import EnvContext
from ray.rllib.utils.annotations import (
from ray.rllib.utils.typing import (
from ray.util import log_once
@PublicAPI
class MultiAgentEnvWrapper(BaseEnv):
    """Internal adapter of MultiAgentEnv to BaseEnv.

    This also supports vectorization if num_envs > 1.
    """

    def __init__(self, make_env: Callable[[int], EnvType], existing_envs: List['MultiAgentEnv'], num_envs: int, restart_failed_sub_environments: bool=False):
        """Wraps MultiAgentEnv(s) into the BaseEnv API.

        Args:
            make_env: Factory that produces a new MultiAgentEnv instance taking the
                vector index as only call argument.
                Must be defined, if the number of existing envs is less than num_envs.
            existing_envs: List of already existing multi-agent envs.
            num_envs: Desired num multiagent envs to have at the end in
                total. This will include the given (already created)
                `existing_envs`.
            restart_failed_sub_environments: If True and any sub-environment (within
                this vectorized env) throws any error during env stepping, we will try
                to restart the faulty sub-environment. This is done
                without disturbing the other (still intact) sub-environments.
        """
        self.make_env = make_env
        self.envs = existing_envs
        self.num_envs = num_envs
        self.restart_failed_sub_environments = restart_failed_sub_environments
        self.terminateds = set()
        self.truncateds = set()
        while len(self.envs) < self.num_envs:
            self.envs.append(self.make_env(len(self.envs)))
        for env in self.envs:
            assert isinstance(env, MultiAgentEnv)
        self._init_env_state(idx=None)
        self._unwrapped_env = self.envs[0].unwrapped

    @override(BaseEnv)
    def poll(self) -> Tuple[MultiEnvDict, MultiEnvDict, MultiEnvDict, MultiEnvDict, MultiEnvDict, MultiEnvDict]:
        obs, rewards, terminateds, truncateds, infos = ({}, {}, {}, {}, {})
        for i, env_state in enumerate(self.env_states):
            obs[i], rewards[i], terminateds[i], truncateds[i], infos[i] = env_state.poll()
        return (obs, rewards, terminateds, truncateds, infos, {})

    @override(BaseEnv)
    def send_actions(self, action_dict: MultiEnvDict) -> None:
        for env_id, agent_dict in action_dict.items():
            if env_id in self.terminateds or env_id in self.truncateds:
                raise ValueError(f'Env {env_id} is already done and cannot accept new actions')
            env = self.envs[env_id]
            try:
                obs, rewards, terminateds, truncateds, infos = env.step(agent_dict)
            except Exception as e:
                if self.restart_failed_sub_environments:
                    logger.exception(e.args[0])
                    self.try_restart(env_id=env_id)
                    obs = e
                    rewards = {}
                    terminateds = {'__all__': True}
                    truncateds = {'__all__': False}
                    infos = {}
                else:
                    raise e
            assert isinstance(obs, (dict, Exception)), 'Not a multi-agent obs dict or an Exception!'
            assert isinstance(rewards, dict), 'Not a multi-agent reward dict!'
            assert isinstance(terminateds, dict), 'Not a multi-agent terminateds dict!'
            assert isinstance(truncateds, dict), 'Not a multi-agent truncateds dict!'
            assert isinstance(infos, dict), 'Not a multi-agent info dict!'
            if isinstance(obs, dict):
                info_diff = set(infos).difference(set(obs))
                if info_diff and info_diff != {'__common__'}:
                    raise ValueError("Key set for infos must be a subset of obs (plus optionally the '__common__' key for infos concerning all/no agents): {} vs {}".format(infos.keys(), obs.keys()))
            if '__all__' not in terminateds:
                raise ValueError("In multi-agent environments, '__all__': True|False must be included in the 'terminateds' dict: got {}.".format(terminateds))
            elif '__all__' not in truncateds:
                raise ValueError("In multi-agent environments, '__all__': True|False must be included in the 'truncateds' dict: got {}.".format(truncateds))
            if terminateds['__all__']:
                self.terminateds.add(env_id)
            if truncateds['__all__']:
                self.truncateds.add(env_id)
            self.env_states[env_id].observe(obs, rewards, terminateds, truncateds, infos)

    @override(BaseEnv)
    def try_reset(self, env_id: Optional[EnvID]=None, *, seed: Optional[int]=None, options: Optional[dict]=None) -> Optional[Tuple[MultiEnvDict, MultiEnvDict]]:
        ret_obs = {}
        ret_infos = {}
        if isinstance(env_id, int):
            env_id = [env_id]
        if env_id is None:
            env_id = list(range(len(self.envs)))
        for idx in env_id:
            obs, infos = self.env_states[idx].reset(seed=seed, options=options)
            if isinstance(obs, Exception):
                if self.restart_failed_sub_environments:
                    self.env_states[idx].env = self.envs[idx] = self.make_env(idx)
                else:
                    raise obs
            else:
                assert isinstance(obs, dict), 'Not a multi-agent obs dict!'
            if obs is not None:
                if idx in self.terminateds:
                    self.terminateds.remove(idx)
                if idx in self.truncateds:
                    self.truncateds.remove(idx)
            ret_obs[idx] = obs
            ret_infos[idx] = infos
        return (ret_obs, ret_infos)

    @override(BaseEnv)
    def try_restart(self, env_id: Optional[EnvID]=None) -> None:
        if isinstance(env_id, int):
            env_id = [env_id]
        if env_id is None:
            env_id = list(range(len(self.envs)))
        for idx in env_id:
            logger.warning(f'Trying to restart sub-environment at index {idx}.')
            self.env_states[idx].env = self.envs[idx] = self.make_env(idx)
            logger.warning(f'Sub-environment at index {idx} restarted successfully.')

    @override(BaseEnv)
    def get_sub_environments(self, as_dict: bool=False) -> Union[Dict[str, EnvType], List[EnvType]]:
        if as_dict:
            return {_id: env_state.env for _id, env_state in enumerate(self.env_states)}
        return [state.env for state in self.env_states]

    @override(BaseEnv)
    def try_render(self, env_id: Optional[EnvID]=None) -> None:
        if env_id is None:
            env_id = 0
        assert isinstance(env_id, int)
        return self.envs[env_id].render()

    @property
    @override(BaseEnv)
    @PublicAPI
    def observation_space(self) -> gym.spaces.Dict:
        return self.envs[0].observation_space

    @property
    @override(BaseEnv)
    @PublicAPI
    def action_space(self) -> gym.Space:
        return self.envs[0].action_space

    @override(BaseEnv)
    def observation_space_contains(self, x: MultiEnvDict) -> bool:
        return all((self.envs[0].observation_space_contains(val) for val in x.values()))

    @override(BaseEnv)
    def action_space_contains(self, x: MultiEnvDict) -> bool:
        return all((self.envs[0].action_space_contains(val) for val in x.values()))

    @override(BaseEnv)
    def observation_space_sample(self, agent_ids: list=None) -> MultiEnvDict:
        return {0: self.envs[0].observation_space_sample(agent_ids)}

    @override(BaseEnv)
    def action_space_sample(self, agent_ids: list=None) -> MultiEnvDict:
        return {0: self.envs[0].action_space_sample(agent_ids)}

    @override(BaseEnv)
    def get_agent_ids(self) -> Set[AgentID]:
        return self.envs[0].get_agent_ids()

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
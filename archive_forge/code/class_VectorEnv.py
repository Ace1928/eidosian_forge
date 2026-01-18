import logging
import gymnasium as gym
import numpy as np
from typing import Callable, List, Optional, Tuple, Union, Set
from ray.rllib.env.base_env import BaseEnv, _DUMMY_AGENT_ID
from ray.rllib.utils.annotations import Deprecated, override, PublicAPI
from ray.rllib.utils.typing import (
from ray.util import log_once
@PublicAPI
class VectorEnv:
    """An environment that supports batch evaluation using clones of sub-envs."""

    def __init__(self, observation_space: gym.Space, action_space: gym.Space, num_envs: int):
        """Initializes a VectorEnv instance.

        Args:
            observation_space: The observation Space of a single
                sub-env.
            action_space: The action Space of a single sub-env.
            num_envs: The number of clones to make of the given sub-env.
        """
        self.observation_space = observation_space
        self.action_space = action_space
        self.num_envs = num_envs

    @staticmethod
    def vectorize_gym_envs(make_env: Optional[Callable[[int], EnvType]]=None, existing_envs: Optional[List[gym.Env]]=None, num_envs: int=1, action_space: Optional[gym.Space]=None, observation_space: Optional[gym.Space]=None, restart_failed_sub_environments: bool=False, env_config=None, policy_config=None) -> '_VectorizedGymEnv':
        """Translates any given gym.Env(s) into a VectorizedEnv object.

        Args:
            make_env: Factory that produces a new gym.Env taking the sub-env's
                vector index as only arg. Must be defined if the
                number of `existing_envs` is less than `num_envs`.
            existing_envs: Optional list of already instantiated sub
                environments.
            num_envs: Total number of sub environments in this VectorEnv.
            action_space: The action space. If None, use existing_envs[0]'s
                action space.
            observation_space: The observation space. If None, use
                existing_envs[0]'s observation space.
            restart_failed_sub_environments: If True and any sub-environment (within
                a vectorized env) throws any error during env stepping, the
                Sampler will try to restart the faulty sub-environment. This is done
                without disturbing the other (still intact) sub-environment and without
                the RolloutWorker crashing.

        Returns:
            The resulting _VectorizedGymEnv object (subclass of VectorEnv).
        """
        return _VectorizedGymEnv(make_env=make_env, existing_envs=existing_envs or [], num_envs=num_envs, observation_space=observation_space, action_space=action_space, restart_failed_sub_environments=restart_failed_sub_environments)

    @PublicAPI
    def vector_reset(self, *, seeds: Optional[List[int]]=None, options: Optional[List[dict]]=None) -> Tuple[List[EnvObsType], List[EnvInfoDict]]:
        """Resets all sub-environments.

        Args:
            seed: The list of seeds to be passed to the sub-environments' when resetting
                them. If None, will not reset any existing PRNGs. If you pass
                integers, the PRNGs will be reset even if they already exists.
            options: The list of options dicts to be passed to the sub-environments'
                when resetting them.

        Returns:
            Tuple consitsing of a list of observations from each environment and
            a list of info dicts from each environment.
        """
        raise NotImplementedError

    @PublicAPI
    def reset_at(self, index: Optional[int]=None, *, seed: Optional[int]=None, options: Optional[dict]=None) -> Union[Tuple[EnvObsType, EnvInfoDict], Exception]:
        """Resets a single sub-environment.

        Args:
            index: An optional sub-env index to reset.
            seed: The seed to be passed to the sub-environment at index `index` when
                resetting it. If None, will not reset any existing PRNG. If you pass an
                integer, the PRNG will be reset even if it already exists.
            options: An options dict to be passed to the sub-environment at index
                `index` when resetting it.

        Returns:
            Tuple consisting of observations from the reset sub environment and
            an info dict of the reset sub environment. Alternatively an Exception
            can be returned, indicating that the reset operation on the sub environment
            has failed (and why it failed).
        """
        raise NotImplementedError

    @PublicAPI
    def restart_at(self, index: Optional[int]=None) -> None:
        """Restarts a single sub-environment.

        Args:
            index: An optional sub-env index to restart.
        """
        raise NotImplementedError

    @PublicAPI
    def vector_step(self, actions: List[EnvActionType]) -> Tuple[List[EnvObsType], List[float], List[bool], List[bool], List[EnvInfoDict]]:
        """Performs a vectorized step on all sub environments using `actions`.

        Args:
            actions: List of actions (one for each sub-env).

        Returns:
            A tuple consisting of
            1) New observations for each sub-env.
            2) Reward values for each sub-env.
            3) Terminated values for each sub-env.
            4) Truncated values for each sub-env.
            5) Info values for each sub-env.
        """
        raise NotImplementedError

    @PublicAPI
    def get_sub_environments(self) -> List[EnvType]:
        """Returns the underlying sub environments.

        Returns:
            List of all underlying sub environments.
        """
        return []

    def try_render_at(self, index: Optional[int]=None) -> Optional[np.ndarray]:
        """Renders a single environment.

        Args:
            index: An optional sub-env index to render.

        Returns:
            Either a numpy RGB image (shape=(w x h x 3) dtype=uint8) or
            None in case rendering is handled directly by this method.
        """
        pass

    @PublicAPI
    def to_base_env(self, make_env: Optional[Callable[[int], EnvType]]=None, num_envs: int=1, remote_envs: bool=False, remote_env_batch_wait_ms: int=0, restart_failed_sub_environments: bool=False) -> 'BaseEnv':
        """Converts an RLlib MultiAgentEnv into a BaseEnv object.

        The resulting BaseEnv is always vectorized (contains n
        sub-environments) to support batched forward passes, where n may
        also be 1. BaseEnv also supports async execution via the `poll` and
        `send_actions` methods and thus supports external simulators.

        Args:
            make_env: A callable taking an int as input (which indicates
                the number of individual sub-environments within the final
                vectorized BaseEnv) and returning one individual
                sub-environment.
            num_envs: The number of sub-environments to create in the
                resulting (vectorized) BaseEnv. The already existing `env`
                will be one of the `num_envs`.
            remote_envs: Whether each sub-env should be a @ray.remote
                actor. You can set this behavior in your config via the
                `remote_worker_envs=True` option.
            remote_env_batch_wait_ms: The wait time (in ms) to poll remote
                sub-environments for, if applicable. Only used if
                `remote_envs` is True.

        Returns:
            The resulting BaseEnv object.
        """
        env = VectorEnvWrapper(self)
        return env

    @Deprecated(new='vectorize_gym_envs', error=True)
    def wrap(self, *args, **kwargs) -> '_VectorizedGymEnv':
        pass

    @Deprecated(new='get_sub_environments', error=True)
    def get_unwrapped(self) -> List[EnvType]:
        pass
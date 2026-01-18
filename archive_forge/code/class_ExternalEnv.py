import gymnasium as gym
import queue
import threading
import uuid
from typing import Callable, Tuple, Optional, TYPE_CHECKING
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.utils.annotations import override, PublicAPI
from ray.rllib.utils.typing import (
from ray.rllib.utils.deprecation import deprecation_warning
@PublicAPI
class ExternalEnv(threading.Thread):
    """An environment that interfaces with external agents.

    Unlike simulator envs, control is inverted: The environment queries the
    policy to obtain actions and in return logs observations and rewards for
    training. This is in contrast to gym.Env, where the algorithm drives the
    simulation through env.step() calls.

    You can use ExternalEnv as the backend for policy serving (by serving HTTP
    requests in the run loop), for ingesting offline logs data (by reading
    offline transitions in the run loop), or other custom use cases not easily
    expressed through gym.Env.

    ExternalEnv supports both on-policy actions (through self.get_action()),
    and off-policy actions (through self.log_action()).

    This env is thread-safe, but individual episodes must be executed serially.

    .. testcode::
        :skipif: True

        from ray.tune import register_env
        from ray.rllib.algorithms.dqn import DQN
        YourExternalEnv = ...
        register_env("my_env", lambda config: YourExternalEnv(config))
        algo = DQN(env="my_env")
        while True:
            print(algo.train())
    """

    @PublicAPI
    def __init__(self, action_space: gym.Space, observation_space: gym.Space, max_concurrent: int=None):
        """Initializes an ExternalEnv instance.

        Args:
            action_space: Action space of the env.
            observation_space: Observation space of the env.
        """
        threading.Thread.__init__(self)
        self.daemon = True
        self.action_space = action_space
        self.observation_space = observation_space
        self._episodes = {}
        self._finished = set()
        self._results_avail_condition = threading.Condition()
        if max_concurrent is not None:
            deprecation_warning('The `max_concurrent` argument has been deprecated. Please configurethe number of episodes using the `rollout_fragment_length` and`batch_mode` arguments. Please raise an issue on the Ray Github if these arguments do not support your expected use case for ExternalEnv', error=True)

    @PublicAPI
    def run(self):
        """Override this to implement the run loop.

        Your loop should continuously:
            1. Call self.start_episode(episode_id)
            2. Call self.[get|log]_action(episode_id, obs, [action]?)
            3. Call self.log_returns(episode_id, reward)
            4. Call self.end_episode(episode_id, obs)
            5. Wait if nothing to do.

        Multiple episodes may be started at the same time.
        """
        raise NotImplementedError

    @PublicAPI
    def start_episode(self, episode_id: Optional[str]=None, training_enabled: bool=True) -> str:
        """Record the start of an episode.

        Args:
            episode_id: Unique string id for the episode or
                None for it to be auto-assigned and returned.
            training_enabled: Whether to use experiences for this
                episode to improve the policy.

        Returns:
            Unique string id for the episode.
        """
        if episode_id is None:
            episode_id = uuid.uuid4().hex
        if episode_id in self._finished:
            raise ValueError('Episode {} has already completed.'.format(episode_id))
        if episode_id in self._episodes:
            raise ValueError('Episode {} is already started'.format(episode_id))
        self._episodes[episode_id] = _ExternalEnvEpisode(episode_id, self._results_avail_condition, training_enabled)
        return episode_id

    @PublicAPI
    def get_action(self, episode_id: str, observation: EnvObsType) -> EnvActionType:
        """Record an observation and get the on-policy action.

        Args:
            episode_id: Episode id returned from start_episode().
            observation: Current environment observation.

        Returns:
            Action from the env action space.
        """
        episode = self._get(episode_id)
        return episode.wait_for_action(observation)

    @PublicAPI
    def log_action(self, episode_id: str, observation: EnvObsType, action: EnvActionType) -> None:
        """Record an observation and (off-policy) action taken.

        Args:
            episode_id: Episode id returned from start_episode().
            observation: Current environment observation.
            action: Action for the observation.
        """
        episode = self._get(episode_id)
        episode.log_action(observation, action)

    @PublicAPI
    def log_returns(self, episode_id: str, reward: float, info: Optional[EnvInfoDict]=None) -> None:
        """Records returns (rewards and infos) from the environment.

        The reward will be attributed to the previous action taken by the
        episode. Rewards accumulate until the next action. If no reward is
        logged before the next action, a reward of 0.0 is assumed.

        Args:
            episode_id: Episode id returned from start_episode().
            reward: Reward from the environment.
            info: Optional info dict.
        """
        episode = self._get(episode_id)
        episode.cur_reward += reward
        if info:
            episode.cur_info = info or {}

    @PublicAPI
    def end_episode(self, episode_id: str, observation: EnvObsType) -> None:
        """Records the end of an episode.

        Args:
            episode_id: Episode id returned from start_episode().
            observation: Current environment observation.
        """
        episode = self._get(episode_id)
        self._finished.add(episode.episode_id)
        episode.done(observation)

    def _get(self, episode_id: str) -> '_ExternalEnvEpisode':
        """Get a started episode by its ID or raise an error."""
        if episode_id in self._finished:
            raise ValueError('Episode {} has already completed.'.format(episode_id))
        if episode_id not in self._episodes:
            raise ValueError('Episode {} not found.'.format(episode_id))
        return self._episodes[episode_id]

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
        if num_envs != 1:
            raise ValueError('External(MultiAgent)Env does not currently support num_envs > 1. One way of solving this would be to treat your Env as a MultiAgentEnv hosting only one type of agent but with several copies.')
        env = ExternalEnvWrapper(self)
        return env
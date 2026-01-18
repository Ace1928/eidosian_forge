from collections import defaultdict
from functools import partial
from typing import List, Tuple
import gymnasium as gym
import numpy as np
import tree  # pip install dm_tree
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.core.models.base import STATE_IN, STATE_OUT
from ray.rllib.env.env_runner import EnvRunner
from ray.rllib.env.wrappers.atari_wrappers import NoopResetEnv, MaxAndSkipEnv
from ray.rllib.env.wrappers.dm_control_wrapper import DMCEnv
from ray.rllib.env.utils import _gym_env_creator
from ray.rllib.evaluation.metrics import RolloutMetrics
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID, SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.env.single_agent_episode import SingleAgentEpisode
from ray.rllib.utils.numpy import one_hot
from ray.tune.registry import ENV_CREATOR, _global_registry
def _sample_timesteps(self, num_timesteps: int, explore: bool=True, random_actions: bool=False, force_reset: bool=False) -> Tuple[List[SingleAgentEpisode], List[SingleAgentEpisode]]:
    """Helper method to run n timesteps.

        See docstring of self.sample() for more details.
        """
    done_episodes_to_return = []
    initial_states = tree.map_structure(lambda s: np.repeat(s, self.num_envs, axis=0), self.module.get_initial_state())
    if force_reset or self._needs_initial_reset:
        obs, _ = self.env.reset()
        self._episodes = [SingleAgentEpisode() for _ in range(self.num_envs)]
        states = initial_states
        is_first = np.ones((self.num_envs,))
        self._needs_initial_reset = False
        for i in range(self.num_envs):
            self._episodes[i].add_env_reset(observation=obs[i])
            self._states[i] = {k: s[i] for k, s in states.items()}
    else:
        obs = np.stack([eps.observations[-1] for eps in self._episodes])
        states = {k: np.stack([initial_states[k][i] if self._states[i] is None else self._states[i][k] for i, eps in enumerate(self._episodes)]) for k in initial_states.keys()}
        is_first = np.zeros((self.num_envs,))
        for i, eps in enumerate(self._episodes):
            if len(eps) == 0:
                is_first[i] = 1.0
    ts = 0
    while ts < num_timesteps:
        if random_actions:
            actions = self.env.action_space.sample()
        else:
            batch = {STATE_IN: tree.map_structure(lambda s: tf.convert_to_tensor(s), states), SampleBatch.OBS: tf.convert_to_tensor(obs), 'is_first': tf.convert_to_tensor(is_first)}
            if explore:
                outs = self.module.forward_exploration(batch)
            else:
                outs = self.module.forward_inference(batch)
            actions = outs[SampleBatch.ACTIONS].numpy()
            if isinstance(self.env.single_action_space, gym.spaces.Discrete):
                actions = np.argmax(actions, axis=-1)
            states = tree.map_structure(lambda s: s.numpy(), outs[STATE_OUT])
        obs, rewards, terminateds, truncateds, infos = self.env.step(actions)
        ts += self.num_envs
        for i in range(self.num_envs):
            s = {k: s[i] for k, s in states.items()}
            if terminateds[i] or truncateds[i]:
                self._episodes[i].add_env_step(observation=infos['final_observation'][i], action=actions[i], reward=rewards[i], terminated=terminateds[i], truncated=truncateds[i])
                self._states[i] = s
                for k, v in self.module.get_initial_state().items():
                    states[k][i] = v.numpy()
                is_first[i] = True
                done_episodes_to_return.append(self._episodes[i])
                self._episodes[i] = SingleAgentEpisode(observations=[obs[i]])
            else:
                self._episodes[i].add_env_step(observation=obs[i], action=actions[i], reward=rewards[i])
                is_first[i] = False
            self._states[i] = s
    self._done_episodes_for_metrics.extend(done_episodes_to_return)
    ongoing_episodes = self._episodes
    self._episodes = [eps.cut() for eps in self._episodes]
    for eps in ongoing_episodes:
        self._ongoing_episodes_for_metrics[eps.id_].append(eps)
    self._ts_since_last_metrics += ts
    return (done_episodes_to_return, ongoing_episodes)
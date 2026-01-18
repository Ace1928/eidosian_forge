import gymnasium as gym
import numpy as np
from typing import Optional
from ray.rllib.env.vector_env import VectorEnv
from ray.rllib.utils.annotations import override
class VectorizedMockEnv(VectorEnv):
    """Vectorized version of the MockEnv.

    Contains `num_envs` MockEnv instances, each one having its own
    `episode_length` horizon.
    """

    def __init__(self, episode_length, num_envs):
        super().__init__(observation_space=gym.spaces.Discrete(1), action_space=gym.spaces.Discrete(2), num_envs=num_envs)
        self.envs = [MockEnv(episode_length) for _ in range(num_envs)]

    @override(VectorEnv)
    def vector_reset(self, *, seeds=None, options=None):
        seeds = seeds or [None] * self.num_envs
        options = options or [None] * self.num_envs
        obs_and_infos = [e.reset(seed=seeds[i], options=options[i]) for i, e in enumerate(self.envs)]
        return ([oi[0] for oi in obs_and_infos], [oi[1] for oi in obs_and_infos])

    @override(VectorEnv)
    def reset_at(self, index, *, seed=None, options=None):
        return self.envs[index].reset(seed=seed, options=options)

    @override(VectorEnv)
    def vector_step(self, actions):
        obs_batch, rew_batch, terminated_batch, truncated_batch, info_batch = ([], [], [], [], [])
        for i in range(len(self.envs)):
            obs, rew, terminated, truncated, info = self.envs[i].step(actions[i])
            obs_batch.append(obs)
            rew_batch.append(rew)
            terminated_batch.append(terminated)
            truncated_batch.append(truncated)
            info_batch.append(info)
        return (obs_batch, rew_batch, terminated_batch, truncated_batch, info_batch)

    @override(VectorEnv)
    def get_sub_environments(self):
        return self.envs
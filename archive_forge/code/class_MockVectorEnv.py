import gymnasium as gym
import numpy as np
from typing import Optional
from ray.rllib.env.vector_env import VectorEnv
from ray.rllib.utils.annotations import override
class MockVectorEnv(VectorEnv):
    """A custom vector env that uses a single(!) CartPole sub-env.

    However, this env pretends to be a vectorized one to illustrate how one
    could create custom VectorEnvs w/o the need for actual vectorizations of
    sub-envs under the hood.
    """

    def __init__(self, episode_length, mocked_num_envs):
        self.env = gym.make('CartPole-v1')
        super().__init__(observation_space=self.env.observation_space, action_space=self.env.action_space, num_envs=mocked_num_envs)
        self.episode_len = episode_length
        self.ts = 0

    @override(VectorEnv)
    def vector_reset(self, *, seeds=None, options=None):
        seeds = seeds or [None]
        options = options or [None]
        obs, infos = self.env.reset(seed=seeds[0], options=options[0])
        return ([obs for _ in range(self.num_envs)], [infos for _ in range(self.num_envs)])

    @override(VectorEnv)
    def reset_at(self, index, *, seed=None, options=None):
        self.ts = 0
        return self.env.reset(seed=seed, options=options)

    @override(VectorEnv)
    def vector_step(self, actions):
        self.ts += 1
        obs_batch, rew_batch, terminated_batch, truncated_batch, info_batch = ([], [], [], [], [])
        for i in range(self.num_envs):
            obs, rew, terminated, truncated, info = self.env.step(actions[i])
            if self.ts >= self.episode_len:
                truncated = True
            obs_batch.append(obs)
            rew_batch.append(rew)
            terminated_batch.append(terminated)
            truncated_batch.append(truncated)
            info_batch.append(info)
            if terminated or truncated:
                remaining = self.num_envs - (i + 1)
                obs_batch.extend([obs for _ in range(remaining)])
                rew_batch.extend([rew for _ in range(remaining)])
                terminated_batch.extend([terminated for _ in range(remaining)])
                truncated_batch.extend([truncated for _ in range(remaining)])
                info_batch.extend([info for _ in range(remaining)])
                break
        return (obs_batch, rew_batch, terminated_batch, truncated_batch, info_batch)

    @override(VectorEnv)
    def get_sub_environments(self):
        return [self.env for _ in range(self.num_envs)]
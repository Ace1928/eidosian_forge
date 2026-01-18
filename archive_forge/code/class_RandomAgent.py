import numpy as np
from typing import Optional
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig, NotProvided
from ray.rllib.utils.annotations import override
from ray.rllib.utils.deprecation import Deprecated, ALGO_DEPRECATION_WARNING
@Deprecated(old='rllib/algorithms/random_agent/', help=ALGO_DEPRECATION_WARNING, error=False)
class RandomAgent(Algorithm):
    """Algo that produces random actions and never learns."""

    @classmethod
    @override(Algorithm)
    def get_default_config(cls) -> AlgorithmConfig:
        config = AlgorithmConfig()
        config.rollouts_per_iteration = 10
        return config

    @override(Algorithm)
    def _init(self, config, env_creator):
        self.env = env_creator(config['env_config'])

    @override(Algorithm)
    def step(self):
        rewards = []
        steps = 0
        for _ in range(self.config.rollouts_per_iteration):
            self.env.reset()
            terminated = truncated = False
            reward = 0.0
            while not terminated and (not truncated):
                action = self.env.action_space.sample()
                _, rew, terminated, truncated, _ = self.env.step(action)
                reward += rew
                steps += 1
            rewards.append(reward)
        return {'episode_reward_mean': np.mean(rewards), 'timesteps_this_iter': steps}
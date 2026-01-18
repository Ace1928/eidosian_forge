import os
import pickle
import time
import numpy as np
from ray.tune import result as tune_result
from ray.rllib.algorithms.algorithm import Algorithm, AlgorithmConfig
from ray.rllib.utils.annotations import override
class _SigmoidFakeData(_MockTrainer):
    """Algorithm that returns sigmoid learning curves.

    This can be helpful for evaluating early stopping algorithms."""

    @classmethod
    @override(Algorithm)
    def get_default_config(cls) -> AlgorithmConfig:
        return AlgorithmConfig().update_from_dict({'width': 100, 'height': 100, 'offset': 0, 'iter_time': 10, 'iter_timesteps': 1})

    def step(self):
        i = max(0, self.iteration - self.config.offset)
        v = np.tanh(float(i) / self.config.width)
        v *= self.config.height
        return dict(episode_reward_mean=v, episode_len_mean=v, timesteps_this_iter=self.config.iter_timesteps, time_this_iter_s=self.config.iter_time, info={})
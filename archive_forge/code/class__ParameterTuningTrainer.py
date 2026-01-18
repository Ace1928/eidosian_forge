import os
import pickle
import time
import numpy as np
from ray.tune import result as tune_result
from ray.rllib.algorithms.algorithm import Algorithm, AlgorithmConfig
from ray.rllib.utils.annotations import override
class _ParameterTuningTrainer(_MockTrainer):

    @classmethod
    @override(Algorithm)
    def get_default_config(cls) -> AlgorithmConfig:
        return AlgorithmConfig().update_from_dict({'reward_amt': 10, 'dummy_param': 10, 'dummy_param2': 15, 'iter_time': 10, 'iter_timesteps': 1})

    def step(self):
        return dict(episode_reward_mean=self.config.reward_amt * self.iteration, episode_len_mean=self.config.reward_amt, timesteps_this_iter=self.config.iter_timesteps, time_this_iter_s=self.config.iter_time, info={})
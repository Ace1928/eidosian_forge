from typing import Union
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.utils.annotations import override
from ray.rllib.utils.deprecation import Deprecated, ALGO_DEPRECATION_WARNING
class BanditConfig(AlgorithmConfig):

    def __init__(self, algo_class: Union['BanditLinTS', 'BanditLinUCB']=None):
        super().__init__(algo_class=algo_class)
        self.framework_str = 'torch'
        self.rollout_fragment_length = 1
        self.train_batch_size = 1
        self.min_sample_timesteps_per_iteration = 100
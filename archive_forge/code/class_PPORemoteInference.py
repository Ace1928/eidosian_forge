import argparse
import os
from typing import Union
import ray
from ray import air, tune
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.utils.annotations import override
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.rllib.utils.typing import PartialAlgorithmConfigDict
from ray.tune import PlacementGroupFactory
from ray.tune.logger import pretty_print
class PPORemoteInference(PPO):

    @classmethod
    @override(Algorithm)
    def default_resource_request(cls, config: Union[AlgorithmConfig, PartialAlgorithmConfigDict]):
        if isinstance(config, AlgorithmConfig):
            cf = config
        else:
            cf = cls.get_default_config().update_from_dict(config)
        return PlacementGroupFactory(bundles=[{'CPU': 1, 'GPU': cf.num_gpus}, {'CPU': cf.num_envs_per_worker}], strategy=cf.placement_strategy)
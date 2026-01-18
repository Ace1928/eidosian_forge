from typing import Optional
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig, NotProvided
from ray.rllib.env.wrappers.model_vector_env import model_vector_env
from ray.rllib.utils.annotations import override
from ray.rllib.utils.deprecation import (
@Deprecated(old='rllib/algorithms/mbmpo/', new='rllib_contrib/mbmpo/', help=ALGO_DEPRECATION_WARNING, error=True)
class MBMPO(Algorithm):

    @classmethod
    @override(Algorithm)
    def get_default_config(cls) -> AlgorithmConfig:
        return MBMPOConfig()
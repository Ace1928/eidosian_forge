from typing import Optional
from ray.rllib.algorithms import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig, NotProvided
from ray.rllib.utils.annotations import override
from ray.rllib.utils.deprecation import Deprecated, ALGO_DEPRECATION_WARNING
@Deprecated(old='rllib/algorithms/es/', new='rllib_contrib/es/', help=ALGO_DEPRECATION_WARNING, error=True)
class ES(Algorithm):

    @classmethod
    @override(Algorithm)
    def get_default_config(cls) -> AlgorithmConfig:
        return ESConfig()
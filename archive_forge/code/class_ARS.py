from typing import Optional
from ray.rllib.algorithms import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig, NotProvided
from ray.rllib.utils.annotations import override
from ray.rllib.utils.deprecation import Deprecated, ALGO_DEPRECATION_WARNING
@Deprecated(old='rllib/algorithms/ars/', new='rllib_contrib/ars/', help=ALGO_DEPRECATION_WARNING, error=True)
class ARS(Algorithm):

    @classmethod
    @override(Algorithm)
    def get_default_config(cls) -> AlgorithmConfig:
        return ARSConfig()
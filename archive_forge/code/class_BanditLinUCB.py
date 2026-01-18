from typing import Union
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.utils.annotations import override
from ray.rllib.utils.deprecation import Deprecated, ALGO_DEPRECATION_WARNING
@Deprecated(old='rllib/algorithms/bandit/', new='rllib_contrib/bandit/', help=ALGO_DEPRECATION_WARNING, error=True)
class BanditLinUCB(Algorithm):

    @classmethod
    @override(Algorithm)
    def get_default_config(cls) -> BanditLinUCBConfig:
        return BanditLinUCBConfig()
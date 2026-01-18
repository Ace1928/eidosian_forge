from typing import List, Optional, Dict, Any, Union
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig, NotProvided
from ray.rllib.utils import deep_update
from ray.rllib.utils.annotations import override
from ray.rllib.utils.deprecation import Deprecated, ALGO_DEPRECATION_WARNING
@Deprecated(old='rllib/algorithms/dt/', new='rllib_contrib/dt/', help=ALGO_DEPRECATION_WARNING, error=True)
class DT(Algorithm):

    @classmethod
    @override(Algorithm)
    def get_default_config(cls) -> AlgorithmConfig:
        return DTConfig()
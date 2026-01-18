from typing import Optional
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig, NotProvided
from ray.rllib.algorithms.dqn import DQN, DQNConfig
from ray.rllib.utils.annotations import override
from ray.rllib.utils.deprecation import (
@Deprecated(old='rllib/algorithms/r2d2/', new='rllib_contrib/r2d2/', help=ALGO_DEPRECATION_WARNING, error=True)
class R2D2(DQN):

    @classmethod
    @override(DQN)
    def get_default_config(cls) -> AlgorithmConfig:
        return R2D2Config()
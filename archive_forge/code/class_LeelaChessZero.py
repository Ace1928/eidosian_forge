from typing import List, Optional, Union
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig, NotProvided
from ray.rllib.utils.annotations import override
from ray.rllib.utils.replay_buffers import PrioritizedReplayBuffer
from ray.rllib.utils.deprecation import (
@Deprecated(old='rllib/algorithms/leela_chess_zero/', new='rllib_contrib/leela_chess_zero/', help=ALGO_DEPRECATION_WARNING, error=True)
class LeelaChessZero(Algorithm):

    @classmethod
    @override(Algorithm)
    def get_default_config(cls) -> AlgorithmConfig:
        return LeelaChessZeroConfig()
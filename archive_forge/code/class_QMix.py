from typing import Optional
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig, NotProvided
from ray.rllib.algorithms.simple_q.simple_q import SimpleQ, SimpleQConfig
from ray.rllib.utils.annotations import override
from ray.rllib.utils.deprecation import (
@Deprecated(old='rllib/algorithms/qmix/', new='rllib_contrib/qmix/', help=ALGO_DEPRECATION_WARNING, error=True)
class QMix(SimpleQ):

    @classmethod
    @override(SimpleQ)
    def get_default_config(cls) -> AlgorithmConfig:
        return QMixConfig()
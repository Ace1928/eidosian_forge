from typing import Optional
from ray._private.dict import merge_dicts
from ray.rllib.algorithms.algorithm_config import NotProvided
from ray.rllib.algorithms.dqn.dqn import DQN, DQNConfig
from ray.rllib.utils.annotations import override
from ray.rllib.utils.deprecation import (
@Deprecated(old='rllib/algorithms/apex_dqn/', new='rllib_contrib/apex_dqn/', help=ALGO_DEPRECATION_WARNING, error=True)
class ApexDQN(DQN):

    @classmethod
    @override(DQN)
    def get_default_config(cls) -> ApexDQNConfig:
        return ApexDQNConfig()
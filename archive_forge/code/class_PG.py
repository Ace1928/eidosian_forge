from typing import List, Optional, Type, Union
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig, NotProvided
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.annotations import override
from ray.rllib.utils.deprecation import Deprecated, ALGO_DEPRECATION_WARNING
@Deprecated(old='rllib/algorithms/pg/', new='rllib_contrib/pg/', help=ALGO_DEPRECATION_WARNING, error=False)
class PG(Algorithm):

    @classmethod
    @override(Algorithm)
    def get_default_config(cls) -> AlgorithmConfig:
        return PGConfig()

    @classmethod
    @override(Algorithm)
    def get_default_policy_class(cls, config: AlgorithmConfig) -> Optional[Type[Policy]]:
        if config['framework'] == 'torch':
            from ray.rllib.algorithms.pg.pg_torch_policy import PGTorchPolicy
            return PGTorchPolicy
        elif config.framework_str == 'tf':
            from ray.rllib.algorithms.pg.pg_tf_policy import PGTF1Policy
            return PGTF1Policy
        else:
            from ray.rllib.algorithms.pg.pg_tf_policy import PGTF2Policy
            return PGTF2Policy
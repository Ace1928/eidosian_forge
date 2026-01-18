from typing import Type, Optional
from ray.rllib.algorithms.sac import (
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig, NotProvided
from ray.rllib.algorithms.sac.rnnsac_torch_policy import RNNSACTorchPolicy
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.annotations import override
from ray.rllib.utils.deprecation import DEPRECATED_VALUE
class RNNSAC(SAC):

    @classmethod
    @override(SAC)
    def get_default_config(cls) -> AlgorithmConfig:
        return RNNSACConfig()

    @classmethod
    @override(SAC)
    def get_default_policy_class(cls, config: AlgorithmConfig) -> Optional[Type[Policy]]:
        return RNNSACTorchPolicy
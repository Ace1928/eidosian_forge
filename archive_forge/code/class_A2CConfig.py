from typing import Optional
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig, NotProvided
from ray.rllib.algorithms.a3c.a3c import A3CConfig, A3C
from ray.rllib.utils.annotations import override
from ray.rllib.utils.deprecation import Deprecated, ALGO_DEPRECATION_WARNING
class A2CConfig(A3CConfig):

    def __init__(self):
        """Initializes a A2CConfig instance."""
        super().__init__(algo_class=A2C)
        self.microbatch_size = None
        self.num_rollout_workers = 2
        self.rollout_fragment_length = 'auto'
        self.sample_async = False
        self.min_time_s_per_iteration = 10

    @override(A3CConfig)
    def training(self, *, microbatch_size: Optional[int]=NotProvided, **kwargs) -> 'A2CConfig':
        super().training(**kwargs)
        if microbatch_size is not NotProvided:
            self.microbatch_size = microbatch_size
        return self
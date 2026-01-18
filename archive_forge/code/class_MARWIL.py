from typing import Callable, Optional, Type, Union
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig, NotProvided
from ray.rllib.execution.rollout_ops import (
from ray.rllib.execution.train_ops import (
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.annotations import override
from ray.rllib.utils.deprecation import deprecation_warning
from ray.rllib.utils.metrics import (
from ray.rllib.utils.typing import (
from ray.tune.logger import Logger
class MARWIL(Algorithm):

    @classmethod
    @override(Algorithm)
    def get_default_config(cls) -> AlgorithmConfig:
        return MARWILConfig()

    @classmethod
    @override(Algorithm)
    def get_default_policy_class(cls, config: AlgorithmConfig) -> Optional[Type[Policy]]:
        if config['framework'] == 'torch':
            from ray.rllib.algorithms.marwil.marwil_torch_policy import MARWILTorchPolicy
            return MARWILTorchPolicy
        elif config['framework'] == 'tf':
            from ray.rllib.algorithms.marwil.marwil_tf_policy import MARWILTF1Policy
            return MARWILTF1Policy
        else:
            from ray.rllib.algorithms.marwil.marwil_tf_policy import MARWILTF2Policy
            return MARWILTF2Policy

    @override(Algorithm)
    def training_step(self) -> ResultDict:
        with self._timers[SAMPLE_TIMER]:
            train_batch = synchronous_parallel_sample(worker_set=self.workers)
        train_batch = train_batch.as_multi_agent()
        self._counters[NUM_AGENT_STEPS_SAMPLED] += train_batch.agent_steps()
        self._counters[NUM_ENV_STEPS_SAMPLED] += train_batch.env_steps()
        if self.config.simple_optimizer:
            train_results = train_one_step(self, train_batch)
        else:
            train_results = multi_gpu_train_one_step(self, train_batch)
        global_vars = {'timestep': self._counters[NUM_AGENT_STEPS_SAMPLED]}
        if self.workers.remote_workers():
            with self._timers[SYNCH_WORKER_WEIGHTS_TIMER]:
                self.workers.sync_weights(policies=list(train_results.keys()), global_vars=global_vars)
        self.workers.local_worker().set_global_vars(global_vars)
        return train_results
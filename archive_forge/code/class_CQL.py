import logging
from typing import Optional, Type
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig, NotProvided
from ray.rllib.algorithms.cql.cql_tf_policy import CQLTFPolicy
from ray.rllib.algorithms.cql.cql_torch_policy import CQLTorchPolicy
from ray.rllib.algorithms.sac.sac import (
from ray.rllib.execution.rollout_ops import (
from ray.rllib.execution.train_ops import (
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.annotations import override
from ray.rllib.utils.deprecation import (
from ray.rllib.utils.framework import try_import_tf, try_import_tfp
from ray.rllib.utils.metrics import (
from ray.rllib.utils.typing import ResultDict
class CQL(SAC):
    """CQL (derived from SAC)."""

    @classmethod
    @override(SAC)
    def get_default_config(cls) -> AlgorithmConfig:
        return CQLConfig()

    @classmethod
    @override(SAC)
    def get_default_policy_class(cls, config: AlgorithmConfig) -> Optional[Type[Policy]]:
        if config['framework'] == 'torch':
            return CQLTorchPolicy
        else:
            return CQLTFPolicy

    @override(SAC)
    def training_step(self) -> ResultDict:
        with self._timers[SAMPLE_TIMER]:
            train_batch = synchronous_parallel_sample(worker_set=self.workers)
        train_batch = train_batch.as_multi_agent()
        self._counters[NUM_AGENT_STEPS_SAMPLED] += train_batch.agent_steps()
        self._counters[NUM_ENV_STEPS_SAMPLED] += train_batch.env_steps()
        post_fn = self.config.get('before_learn_on_batch') or (lambda b, *a: b)
        train_batch = post_fn(train_batch, self.workers, self.config)
        if self.config.get('simple_optimizer') is True:
            train_results = train_one_step(self, train_batch)
        else:
            train_results = multi_gpu_train_one_step(self, train_batch)
        cur_ts = self._counters[NUM_AGENT_STEPS_TRAINED if self.config.count_steps_by == 'agent_steps' else NUM_ENV_STEPS_TRAINED]
        last_update = self._counters[LAST_TARGET_UPDATE_TS]
        if cur_ts - last_update >= self.config.target_network_update_freq:
            with self._timers[TARGET_NET_UPDATE_TIMER]:
                to_update = self.workers.local_worker().get_policies_to_train()
                self.workers.local_worker().foreach_policy_to_train(lambda p, pid: pid in to_update and p.update_target())
            self._counters[NUM_TARGET_UPDATES] += 1
            self._counters[LAST_TARGET_UPDATE_TS] = cur_ts
        if self.workers.num_remote_workers() > 0:
            with self._timers[SYNCH_WORKER_WEIGHTS_TIMER]:
                self.workers.sync_weights(policies=list(train_results.keys()))
        return train_results
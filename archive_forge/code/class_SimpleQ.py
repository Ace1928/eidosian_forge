import logging
from typing import List, Optional, Type, Union
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig, NotProvided
from ray.rllib.algorithms.simple_q.simple_q_tf_policy import (
from ray.rllib.algorithms.simple_q.simple_q_torch_policy import SimpleQTorchPolicy
from ray.rllib.execution.rollout_ops import synchronous_parallel_sample
from ray.rllib.execution.train_ops import multi_gpu_train_one_step, train_one_step
from ray.rllib.policy.policy import Policy
from ray.rllib.utils import deep_update
from ray.rllib.utils.annotations import override
from ray.rllib.utils.deprecation import (
from ray.rllib.utils.metrics import (
from ray.rllib.utils.replay_buffers.utils import (
from ray.rllib.utils.typing import ResultDict
@Deprecated(old='rllib/algorithms/simple_q/', new='rllib_contrib/simple_q/', help=ALGO_DEPRECATION_WARNING, error=False)
class SimpleQ(Algorithm):

    @classmethod
    @override(Algorithm)
    def get_default_config(cls) -> AlgorithmConfig:
        return SimpleQConfig()

    @classmethod
    @override(Algorithm)
    def get_default_policy_class(cls, config: AlgorithmConfig) -> Optional[Type[Policy]]:
        if config['framework'] == 'torch':
            return SimpleQTorchPolicy
        elif config['framework'] == 'tf':
            return SimpleQTF1Policy
        else:
            return SimpleQTF2Policy

    @override(Algorithm)
    def training_step(self) -> ResultDict:
        """Simple Q training iteration function.

        Simple Q consists of the following steps:
        - Sample n MultiAgentBatches from n workers synchronously.
        - Store new samples in the replay buffer.
        - Sample one training MultiAgentBatch from the replay buffer.
        - Learn on the training batch.
        - Update the target network every `target_network_update_freq` sample steps.
        - Return all collected training metrics for the iteration.

        Returns:
            The results dict from executing the training iteration.
        """
        batch_size = self.config.train_batch_size
        local_worker = self.workers.local_worker()
        with self._timers[SAMPLE_TIMER]:
            new_sample_batches = synchronous_parallel_sample(worker_set=self.workers, concat=False)
        for batch in new_sample_batches:
            self._counters[NUM_ENV_STEPS_SAMPLED] += batch.env_steps()
            self._counters[NUM_AGENT_STEPS_SAMPLED] += batch.agent_steps()
            self.local_replay_buffer.add(batch)
        global_vars = {'timestep': self._counters[NUM_ENV_STEPS_SAMPLED]}
        cur_ts = self._counters[NUM_AGENT_STEPS_SAMPLED if self.config.count_steps_by == 'agent_steps' else NUM_ENV_STEPS_SAMPLED]
        if cur_ts > self.config.num_steps_sampled_before_learning_starts:
            train_batch = self.local_replay_buffer.sample(batch_size)
            if self.config.get('simple_optimizer') is True:
                train_results = train_one_step(self, train_batch)
            else:
                train_results = multi_gpu_train_one_step(self, train_batch)
            update_priorities_in_replay_buffer(self.local_replay_buffer, self.config, train_batch, train_results)
            last_update = self._counters[LAST_TARGET_UPDATE_TS]
            if cur_ts - last_update >= self.config.target_network_update_freq:
                with self._timers[TARGET_NET_UPDATE_TIMER]:
                    to_update = local_worker.get_policies_to_train()
                    local_worker.foreach_policy_to_train(lambda p, pid: pid in to_update and p.update_target())
                self._counters[NUM_TARGET_UPDATES] += 1
                self._counters[LAST_TARGET_UPDATE_TS] = cur_ts
            with self._timers[SYNCH_WORKER_WEIGHTS_TIMER]:
                self.workers.sync_weights(policies=list(train_results.keys()), global_vars=global_vars)
        else:
            train_results = {}
        return train_results
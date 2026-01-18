import logging
from typing import List, Optional, Type, Callable
import numpy as np
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig, NotProvided
from ray.rllib.algorithms.dqn.dqn_tf_policy import DQNTFPolicy
from ray.rllib.algorithms.dqn.dqn_torch_policy import DQNTorchPolicy
from ray.rllib.algorithms.simple_q.simple_q import (
from ray.rllib.execution.rollout_ops import (
from ray.rllib.policy.sample_batch import MultiAgentBatch
from ray.rllib.execution.train_ops import (
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.annotations import override
from ray.rllib.utils.replay_buffers.utils import update_priorities_in_replay_buffer
from ray.rllib.utils.typing import ResultDict
from ray.rllib.utils.metrics import (
from ray.rllib.utils.deprecation import DEPRECATED_VALUE
from ray.rllib.utils.replay_buffers.utils import sample_min_n_steps_from_buffer
class DQN(SimpleQ):

    @classmethod
    @override(SimpleQ)
    def get_default_config(cls) -> AlgorithmConfig:
        return DQNConfig()

    @classmethod
    @override(SimpleQ)
    def get_default_policy_class(cls, config: AlgorithmConfig) -> Optional[Type[Policy]]:
        if config['framework'] == 'torch':
            return DQNTorchPolicy
        else:
            return DQNTFPolicy

    @override(SimpleQ)
    def training_step(self) -> ResultDict:
        """DQN training iteration function.

        Each training iteration, we:
        - Sample (MultiAgentBatch) from workers.
        - Store new samples in replay buffer.
        - Sample training batch (MultiAgentBatch) from replay buffer.
        - Learn on training batch.
        - Update remote workers' new policy weights.
        - Update target network every `target_network_update_freq` sample steps.
        - Return all collected metrics for the iteration.

        Returns:
            The results dict from executing the training iteration.
        """
        train_results = {}
        store_weight, sample_and_train_weight = calculate_rr_weights(self.config)
        for _ in range(store_weight):
            with self._timers[SAMPLE_TIMER]:
                new_sample_batch = synchronous_parallel_sample(worker_set=self.workers, concat=True)
            self._counters[NUM_AGENT_STEPS_SAMPLED] += new_sample_batch.agent_steps()
            self._counters[NUM_ENV_STEPS_SAMPLED] += new_sample_batch.env_steps()
            self.local_replay_buffer.add(new_sample_batch)
        global_vars = {'timestep': self._counters[NUM_ENV_STEPS_SAMPLED]}
        cur_ts = self._counters[NUM_AGENT_STEPS_SAMPLED if self.config.count_steps_by == 'agent_steps' else NUM_ENV_STEPS_SAMPLED]
        if cur_ts > self.config.num_steps_sampled_before_learning_starts:
            for _ in range(sample_and_train_weight):
                train_batch = sample_min_n_steps_from_buffer(self.local_replay_buffer, self.config.train_batch_size, count_by_agent_steps=self.config.count_steps_by == 'agent_steps')
                post_fn = self.config.get('before_learn_on_batch') or (lambda b, *a: b)
                train_batch = post_fn(train_batch, self.workers, self.config)
                if self.config.get('simple_optimizer') is True:
                    train_results = train_one_step(self, train_batch)
                else:
                    train_results = multi_gpu_train_one_step(self, train_batch)
                update_priorities_in_replay_buffer(self.local_replay_buffer, self.config, train_batch, train_results)
                last_update = self._counters[LAST_TARGET_UPDATE_TS]
                if cur_ts - last_update >= self.config.target_network_update_freq:
                    to_update = self.workers.local_worker().get_policies_to_train()
                    self.workers.local_worker().foreach_policy_to_train(lambda p, pid: pid in to_update and p.update_target())
                    self._counters[NUM_TARGET_UPDATES] += 1
                    self._counters[LAST_TARGET_UPDATE_TS] = cur_ts
                with self._timers[SYNCH_WORKER_WEIGHTS_TIMER]:
                    self.workers.sync_weights(global_vars=global_vars)
        return train_results
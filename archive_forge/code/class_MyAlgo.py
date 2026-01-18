import argparse
import os
import ray
from ray import air, tune
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.algorithms.dqn.dqn import DQNConfig
from ray.rllib.algorithms.dqn.dqn_tf_policy import DQNTFPolicy
from ray.rllib.algorithms.dqn.dqn_torch_policy import DQNTorchPolicy
from ray.rllib.algorithms.ppo.ppo import PPOConfig
from ray.rllib.algorithms.ppo.ppo_tf_policy import PPOTF1Policy
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.execution.rollout_ops import synchronous_parallel_sample
from ray.rllib.execution.train_ops import train_one_step
from ray.rllib.utils.replay_buffers.multi_agent_replay_buffer import (
from ray.rllib.examples.env.multi_agent import MultiAgentCartPole
from ray.rllib.policy.sample_batch import MultiAgentBatch, concat_samples
from ray.rllib.utils.annotations import override
from ray.rllib.utils.metrics import (
from ray.rllib.utils.sgd import standardized
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.rllib.utils.typing import ResultDict
from ray.tune.registry import register_env
class MyAlgo(Algorithm):

    @override(Algorithm)
    def setup(self, config):
        super().setup(config)
        self.local_replay_buffer = MultiAgentReplayBuffer(num_shards=1, capacity=50000)

    @override(Algorithm)
    def training_step(self) -> ResultDict:
        ppo_batches = []
        num_env_steps = 0
        while num_env_steps < 200:
            ma_batches = synchronous_parallel_sample(worker_set=self.workers, concat=False)
            for ma_batch in ma_batches:
                self._counters[NUM_ENV_STEPS_SAMPLED] += ma_batch.count
                self._counters[NUM_AGENT_STEPS_SAMPLED] += ma_batch.agent_steps()
                ppo_batch = ma_batch.policy_batches.pop('ppo_policy')
                self.local_replay_buffer.add(ma_batch)
                ppo_batches.append(ppo_batch)
                num_env_steps += ppo_batch.count
        dqn_train_results = {}
        if self._counters[NUM_ENV_STEPS_SAMPLED] > 1000:
            for _ in range(10):
                dqn_train_batch = self.local_replay_buffer.sample(num_items=64)
                dqn_train_results = train_one_step(self, dqn_train_batch, ['dqn_policy'])
                self._counters['agent_steps_trained_DQN'] += dqn_train_batch.agent_steps()
                print('DQN policy learning on samples from', 'agent steps trained', dqn_train_batch.agent_steps())
        if self._counters['agent_steps_trained_DQN'] - self._counters[LAST_TARGET_UPDATE_TS] >= self.get_policy('dqn_policy').config['target_network_update_freq']:
            self.workers.local_worker().get_policy('dqn_policy').update_target()
            self._counters[NUM_TARGET_UPDATES] += 1
            self._counters[LAST_TARGET_UPDATE_TS] = self._counters['agent_steps_trained_DQN']
        ppo_train_batch = concat_samples(ppo_batches)
        self._counters['agent_steps_trained_PPO'] += ppo_train_batch.agent_steps()
        ppo_train_batch[Postprocessing.ADVANTAGES] = standardized(ppo_train_batch[Postprocessing.ADVANTAGES])
        print('PPO policy learning on samples from', 'agent steps trained', ppo_train_batch.agent_steps())
        ppo_train_batch = MultiAgentBatch({'ppo_policy': ppo_train_batch}, ppo_train_batch.count)
        ppo_train_results = train_one_step(self, ppo_train_batch, ['ppo_policy'])
        results = dict(ppo_train_results, **dqn_train_results)
        return results
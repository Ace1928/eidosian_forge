import dataclasses
import logging
from typing import List, Optional, Type, Union, TYPE_CHECKING
import numpy as np
import tree
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig, NotProvided
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.rllib.algorithms.ppo.ppo_learner import (
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.evaluation.postprocessing_v2 import postprocess_episodes_to_sample_batch
from ray.rllib.evaluation.rollout_worker import RolloutWorker
from ray.rllib.execution.rollout_ops import (
from ray.rllib.execution.train_ops import (
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.annotations import override
from ray.rllib.utils.deprecation import DEPRECATED_VALUE
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY
from ray.rllib.utils.metrics import (
from ray.rllib.env.single_agent_episode import SingleAgentEpisode
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.schedules.scheduler import Scheduler
from ray.rllib.utils.typing import ResultDict
from ray.util.debug import log_once
class PPO(Algorithm):

    @classmethod
    @override(Algorithm)
    def get_default_config(cls) -> AlgorithmConfig:
        return PPOConfig()

    @classmethod
    @override(Algorithm)
    def get_default_policy_class(cls, config: AlgorithmConfig) -> Optional[Type[Policy]]:
        if config['framework'] == 'torch':
            from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
            return PPOTorchPolicy
        elif config['framework'] == 'tf':
            from ray.rllib.algorithms.ppo.ppo_tf_policy import PPOTF1Policy
            return PPOTF1Policy
        else:
            from ray.rllib.algorithms.ppo.ppo_tf_policy import PPOTF2Policy
            return PPOTF2Policy

    @override(Algorithm)
    def training_step(self) -> ResultDict:
        use_rollout_worker = self.config.env_runner_cls is None or issubclass(self.config.env_runner_cls, RolloutWorker)
        with self._timers[SAMPLE_TIMER]:
            if use_rollout_worker:
                if self.config.count_steps_by == 'agent_steps':
                    train_batch = synchronous_parallel_sample(worker_set=self.workers, max_agent_steps=self.config.train_batch_size)
                else:
                    train_batch = synchronous_parallel_sample(worker_set=self.workers, max_env_steps=self.config.train_batch_size)
            else:
                if self.workers.num_remote_workers() <= 0:
                    episodes: List[SingleAgentEpisode] = [self.workers.local_worker().sample()]
                else:
                    episodes: List[SingleAgentEpisode] = self.workers.foreach_worker(lambda w: w.sample(), local_worker=False)
                postprocessed_episodes: List[SingleAgentEpisode] = self.postprocess_episodes(tree.flatten(episodes))
                train_batch: SampleBatch = postprocess_episodes_to_sample_batch(postprocessed_episodes)
        train_batch = train_batch.as_multi_agent()
        self._counters[NUM_AGENT_STEPS_SAMPLED] += train_batch.agent_steps()
        self._counters[NUM_ENV_STEPS_SAMPLED] += train_batch.env_steps()
        train_batch = standardize_fields(train_batch, ['advantages'])
        if self.config._enable_new_api_stack:
            if self.config.env_runner_cls is None or self.config.env_runner_cls.__name__ == 'RolloutWorker':
                is_module_trainable = self.workers.local_worker().is_policy_to_train
                self.learner_group.set_is_module_trainable(is_module_trainable)
            train_results = self.learner_group.update(train_batch, minibatch_size=self.config.sgd_minibatch_size, num_iters=self.config.num_sgd_iter)
        elif self.config.simple_optimizer:
            train_results = train_one_step(self, train_batch)
        else:
            train_results = multi_gpu_train_one_step(self, train_batch)
        if self.config._enable_new_api_stack:
            policies_to_update = set(train_results.keys()) - {ALL_MODULES}
        else:
            policies_to_update = list(train_results.keys())
        if not use_rollout_worker:
            global_vars = None
        else:
            global_vars = {'timestep': self._counters[NUM_AGENT_STEPS_SAMPLED], 'num_grad_updates_per_policy': {pid: self.workers.local_worker().policy_map[pid].num_grad_updates for pid in policies_to_update}}
        with self._timers[SYNCH_WORKER_WEIGHTS_TIMER]:
            if self.workers.num_remote_workers() > 0:
                from_worker_or_learner_group = None
                if self.config._enable_new_api_stack:
                    from_worker_or_learner_group = self.learner_group
                self.workers.sync_weights(from_worker_or_learner_group=from_worker_or_learner_group, policies=policies_to_update, global_vars=global_vars)
            elif self.config._enable_new_api_stack:
                weights = self.learner_group.get_weights()
                self.workers.local_worker().set_weights(weights)
        if self.config._enable_new_api_stack:
            kl_dict = {}
            if self.config.use_kl_loss:
                for pid in policies_to_update:
                    kl = train_results[pid][LEARNER_RESULTS_KL_KEY]
                    kl_dict[pid] = kl
                    if np.isnan(kl):
                        logger.warning(f'KL divergence for Module {pid} is non-finite, this will likely destabilize your model and the training process. Action(s) in a specific state have near-zero probability. This can happen naturally in deterministic environments where the optimal policy has zero mass for a specific action. To fix this issue, consider setting `kl_coeff` to 0.0 or increasing `entropy_coeff` in your config.')
            additional_results = self.learner_group.additional_update(module_ids_to_update=policies_to_update, sampled_kl_values=kl_dict, timestep=self._counters[NUM_AGENT_STEPS_SAMPLED])
            for pid, res in additional_results.items():
                train_results[pid].update(res)
            return train_results
        for policy_id, policy_info in train_results.items():
            kl_divergence = policy_info[LEARNER_STATS_KEY].get('kl')
            self.get_policy(policy_id).update_kl(kl_divergence)
            scaled_vf_loss = self.config.vf_loss_coeff * policy_info[LEARNER_STATS_KEY]['vf_loss']
            policy_loss = policy_info[LEARNER_STATS_KEY]['policy_loss']
            if log_once('ppo_warned_lr_ratio') and self.config.get('model', {}).get('vf_share_layers') and (scaled_vf_loss > 100):
                logger.warning('The magnitude of your value function loss for policy: {} is extremely large ({}) compared to the policy loss ({}). This can prevent the policy from learning. Consider scaling down the VF loss by reducing vf_loss_coeff, or disabling vf_share_layers.'.format(policy_id, scaled_vf_loss, policy_loss))
            train_batch.policy_batches[policy_id].set_get_interceptor(None)
            mean_reward = train_batch.policy_batches[policy_id]['rewards'].mean()
            if log_once('ppo_warned_vf_clip') and mean_reward > self.config.vf_clip_param:
                self.warned_vf_clip = True
                logger.warning(f'The mean reward returned from the environment is {mean_reward} but the vf_clip_param is set to {self.config['vf_clip_param']}. Consider increasing it for policy: {policy_id} to improve value function convergence.')
        self.workers.local_worker().set_global_vars(global_vars)
        return train_results

    def postprocess_episodes(self, episodes: List[SingleAgentEpisode]) -> List[SingleAgentEpisode]:
        """Calculate advantages and value targets."""
        from ray.rllib.evaluation.postprocessing_v2 import compute_gae_for_episode
        postprocessed_episodes = []
        for episode in episodes:
            postprocessed_episodes.append(compute_gae_for_episode(episode, self.config, self.workers.local_worker().module))
        return postprocessed_episodes
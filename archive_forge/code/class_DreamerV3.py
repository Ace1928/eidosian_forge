import copy
import dataclasses
import gc
import logging
import tree  # pip install dm_tree
from typing import Any, Dict, List, Optional, Union
import gymnasium as gym
import numpy as np
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig, NotProvided
from ray.rllib.algorithms.dreamerv3.dreamerv3_catalog import DreamerV3Catalog
from ray.rllib.algorithms.dreamerv3.dreamerv3_learner import (
from ray.rllib.algorithms.dreamerv3.utils import do_symlog_obs
from ray.rllib.algorithms.dreamerv3.utils.env_runner import DreamerV3EnvRunner
from ray.rllib.algorithms.dreamerv3.utils.summaries import (
from ray.rllib.core.learner.learner import LearnerHyperparameters
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.models.catalog import MODEL_DEFAULTS
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID, SampleBatch
from ray.rllib.utils import deep_update
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.numpy import one_hot
from ray.rllib.utils.metrics import (
from ray.rllib.utils.replay_buffers.episode_replay_buffer import EpisodeReplayBuffer
from ray.rllib.utils.typing import LearningRateOrSchedule, ResultDict
class DreamerV3(Algorithm):
    """Implementation of the model-based DreamerV3 RL algorithm described in [1]."""

    @override(Algorithm)
    def compute_single_action(self, *args, **kwargs):
        raise NotImplementedError('DreamerV3 does not support the `compute_single_action()` API. Refer to the README here (https://github.com/ray-project/ray/tree/master/rllib/algorithms/dreamerv3) to find more information on how to run action inference with this algorithm.')

    @classmethod
    @override(Algorithm)
    def get_default_config(cls) -> AlgorithmConfig:
        return DreamerV3Config()

    @override(Algorithm)
    def setup(self, config: AlgorithmConfig):
        super().setup(config)
        if self.config.share_module_between_env_runner_and_learner:
            assert self.workers.local_worker().module is None
            self.workers.local_worker().module = self.learner_group._learner.module[DEFAULT_POLICY_ID]
        if self.config.framework_str == 'tf2':
            self.workers.local_worker().module.dreamer_model.summary(expand_nested=True)
        self.replay_buffer = EpisodeReplayBuffer(capacity=self.config.replay_buffer_config['capacity'], batch_size_B=self.config.batch_size_B, batch_length_T=self.config.batch_length_T)

    @override(Algorithm)
    def training_step(self) -> ResultDict:
        results = {}
        env_runner = self.workers.local_worker()
        if self.training_iteration == 0:
            logger.info(f'Filling replay buffer so it contains at least {self.config.batch_size_B * self.config.batch_length_T} timesteps (required for a single train batch).')
        have_sampled = False
        with self._timers[SAMPLE_TIMER]:
            while self.replay_buffer.get_num_timesteps() < self.config.batch_size_B * self.config.batch_length_T or self.training_ratio >= self.config.training_ratio or (not have_sampled):
                done_episodes, ongoing_episodes = env_runner.sample()
                self.replay_buffer.add(episodes=done_episodes + ongoing_episodes)
                have_sampled = True
                env_steps_last_regular_sample = sum((len(eps) for eps in done_episodes + ongoing_episodes))
                total_sampled = env_steps_last_regular_sample
                if self._counters[NUM_AGENT_STEPS_SAMPLED] == 0:
                    d_, o_ = env_runner.sample(num_timesteps=self.config.batch_size_B * self.config.batch_length_T - env_steps_last_regular_sample, random_actions=True)
                    self.replay_buffer.add(episodes=d_ + o_)
                    total_sampled += sum((len(eps) for eps in d_ + o_))
                self._counters[NUM_AGENT_STEPS_SAMPLED] += total_sampled
                self._counters[NUM_ENV_STEPS_SAMPLED] += total_sampled
        results[ALL_MODULES] = report_sampling_and_replay_buffer(replay_buffer=self.replay_buffer)
        replayed_steps_this_iter = sub_iter = 0
        while replayed_steps_this_iter / env_steps_last_regular_sample < self.config.training_ratio:
            with self._timers[LEARN_ON_BATCH_TIMER]:
                logger.info(f'\tSub-iteration {self.training_iteration}/{sub_iter})')
                sample = self.replay_buffer.sample(batch_size_B=self.config.batch_size_B, batch_length_T=self.config.batch_length_T)
                replayed_steps = self.config.batch_size_B * self.config.batch_length_T
                replayed_steps_this_iter += replayed_steps
                if isinstance(env_runner.env.single_action_space, gym.spaces.Discrete):
                    sample['actions_ints'] = sample[SampleBatch.ACTIONS]
                    sample[SampleBatch.ACTIONS] = one_hot(sample['actions_ints'], depth=env_runner.env.single_action_space.n)
                train_results = self.learner_group.update(SampleBatch(sample).as_multi_agent(), reduce_fn=self._reduce_results)
                self._counters[NUM_AGENT_STEPS_TRAINED] += replayed_steps
                self._counters[NUM_ENV_STEPS_TRAINED] += replayed_steps
                with self._timers['critic_ema_update']:
                    self.learner_group.additional_update(timestep=self._counters[NUM_ENV_STEPS_SAMPLED], reduce_fn=self._reduce_results)
                if self.config.report_images_and_videos:
                    report_predicted_vs_sampled_obs(results=train_results[DEFAULT_POLICY_ID], sample=sample, batch_size_B=self.config.batch_size_B, batch_length_T=self.config.batch_length_T, symlog_obs=do_symlog_obs(env_runner.env.single_observation_space, self.config.symlog_obs))
                res = train_results[DEFAULT_POLICY_ID]
                logger.info(f'\t\tWORLD_MODEL_L_total={res['WORLD_MODEL_L_total']:.5f} (L_pred={res['WORLD_MODEL_L_prediction']:.5f} (decoder/obs={res['WORLD_MODEL_L_decoder']} L_rew={res['WORLD_MODEL_L_reward']} L_cont={res['WORLD_MODEL_L_continue']}); L_dyn/rep={res['WORLD_MODEL_L_dynamics']:.5f})')
                msg = '\t\t'
                if self.config.train_actor:
                    msg += f'L_actor={res['ACTOR_L_total']:.5f} '
                if self.config.train_critic:
                    msg += f'L_critic={res['CRITIC_L_total']:.5f} '
                logger.info(msg)
                sub_iter += 1
                self._counters[NUM_GRAD_UPDATES_LIFETIME] += 1
        with self._timers[SYNCH_WORKER_WEIGHTS_TIMER]:
            if not self.config.share_module_between_env_runner_and_learner:
                self._counters[NUM_TRAINING_STEP_CALLS_SINCE_LAST_SYNCH_WORKER_WEIGHTS] = 0
                self._counters[NUM_SYNCH_WORKER_WEIGHTS] += 1
                self.workers.sync_weights(from_worker_or_learner_group=self.learner_group)
        if self.config.gc_frequency_train_steps and self.training_iteration % self.config.gc_frequency_train_steps == 0:
            with self._timers[GARBAGE_COLLECTION_TIMER]:
                gc.collect()
        results.update(train_results)
        results[ALL_MODULES]['actual_training_ratio'] = self.training_ratio
        return results

    @property
    def training_ratio(self) -> float:
        """Returns the actual training ratio of this Algorithm.

        The training ratio is copmuted by dividing the total number of steps
        trained thus far (replayed from the buffer) over the total number of actual
        env steps taken thus far.
        """
        return self._counters[NUM_ENV_STEPS_TRAINED] / self._counters[NUM_ENV_STEPS_SAMPLED]

    @staticmethod
    def _reduce_results(results: List[Dict[str, Any]]):
        return tree.map_structure(lambda *s: np.mean(s, axis=0), *results)
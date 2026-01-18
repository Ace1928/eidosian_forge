from collections import defaultdict
import concurrent
import copy
from datetime import datetime
import functools
import gymnasium as gym
import importlib
import json
import logging
import numpy as np
import os
from packaging import version
import pkg_resources
import re
import tempfile
import time
import tree  # pip install dm_tree
from typing import (
import ray
from ray._private.usage.usage_lib import TagKey, record_extra_usage_tag
from ray.actor import ActorHandle
from ray.train import Checkpoint
import ray.cloudpickle as pickle
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.algorithms.registry import ALGORITHMS_CLASS_TO_NAME as ALL_ALGORITHMS
from ray.rllib.connectors.agent.obs_preproc import ObsPreprocessorConnector
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.env.env_context import EnvContext
from ray.rllib.env.env_runner import EnvRunner
from ray.rllib.env.utils import _gym_env_creator
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.metrics import (
from ray.rllib.evaluation.postprocessing_v2 import postprocess_episodes_to_sample_batch
from ray.rllib.evaluation.rollout_worker import RolloutWorker
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.execution.rollout_ops import synchronous_parallel_sample
from ray.rllib.execution.train_ops import multi_gpu_train_one_step, train_one_step
from ray.rllib.offline import get_dataset_and_shards
from ray.rllib.offline.estimators import (
from ray.rllib.offline.offline_evaluator import OfflineEvaluator
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID, SampleBatch, concat_samples
from ray.rllib.utils import deep_update, FilterManager
from ray.rllib.utils.annotations import (
from ray.rllib.utils.checkpoints import (
from ray.rllib.utils.debug import update_global_seed_if_necessary
from ray.rllib.utils.deprecation import (
from ray.rllib.utils.error import ERR_MSG_INVALID_ENV_DESCRIPTOR, EnvError
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.from_config import from_config
from ray.rllib.utils.metrics import (
from ray.rllib.utils.metrics.learner_info import LEARNER_INFO
from ray.rllib.utils.policy import validate_policy_id
from ray.rllib.utils.replay_buffers import MultiAgentReplayBuffer, ReplayBuffer
from ray.rllib.utils.serialization import deserialize_type, NOT_SERIALIZABLE
from ray.rllib.utils.spaces import space_utils
from ray.rllib.utils.typing import (
from ray.tune.execution.placement_groups import PlacementGroupFactory
from ray.tune.experiment.trial import ExportFormat
from ray.tune.logger import Logger, UnifiedLogger
from ray.tune.registry import ENV_CREATOR, _global_registry
from ray.tune.resources import Resources
from ray.tune.result import DEFAULT_RESULTS_DIR
from ray.tune.trainable import Trainable
from ray.util import log_once
from ray.util.timer import _Timer
from ray.tune.registry import get_trainable_cls
@ExperimentalAPI
def _evaluate_async(self, duration_fn: Optional[Callable[[int], int]]=None) -> dict:
    """Evaluates current policy under `evaluation_config` settings.

        Uses the AsyncParallelRequests manager to send frequent `sample.remote()`
        requests to the evaluation EnvRunners and collect the results of these
        calls. Handles worker failures (or slowdowns) gracefully due to the asynch'ness
        and the fact that other eval EnvRunners can thus cover the workload.

        Important Note: This will replace the current `self.evaluate()` method as the
        default in the future.

        Args:
            duration_fn: An optional callable taking the already run
                num episodes as only arg and returning the number of
                episodes left to run. It's used to find out whether
                evaluation should continue.
        """
    unit = self.config.evaluation_duration_unit
    eval_cfg = self.evaluation_config
    rollout = eval_cfg.rollout_fragment_length
    num_envs = eval_cfg.num_envs_per_worker
    auto = self.config.evaluation_duration == 'auto'
    duration = self.config.evaluation_duration if not auto else (self.config.evaluation_num_workers or 1) * (1 if unit == 'episodes' else rollout)
    self._before_evaluate()
    self._sync_filters_if_needed(central_worker=self.workers.local_worker(), workers=self.evaluation_workers, config=eval_cfg)
    if self.config.custom_evaluation_function:
        raise ValueError('`config.custom_evaluation_function` not supported in combination with `enable_async_evaluation=True` config setting!')
    if self.evaluation_workers is None and (self.workers.local_worker().input_reader is None or self.config.evaluation_num_workers == 0):
        raise ValueError('Evaluation w/o eval workers (calling Algorithm.evaluate() w/o evaluation specifically set up) OR evaluation without input reader OR evaluation with only a local evaluation worker (`evaluation_num_workers=0`) not supported in combination with `enable_async_evaluation=True` config setting!')
    agent_steps_this_iter = 0
    env_steps_this_iter = 0
    logger.info(f'Evaluating current state of {self} for {duration} {unit}.')
    all_batches = []
    if duration_fn is None:

        def duration_fn(num_units_done):
            return duration - num_units_done
    self._evaluation_weights_seq_number += 1
    weights_ref = ray.put(self.workers.local_worker().get_weights())
    weights_seq_no = self._evaluation_weights_seq_number

    def remote_fn(worker):
        worker.set_weights(weights=ray.get(weights_ref), weights_seq_no=weights_seq_no)
        batch = worker.sample()
        metrics = worker.get_metrics()
        return (batch, metrics, weights_seq_no)
    rollout_metrics = []
    num_units_done = 0
    _round = 0
    while self.evaluation_workers.num_healthy_remote_workers() > 0:
        units_left_to_do = duration_fn(num_units_done)
        if units_left_to_do <= 0:
            break
        _round += 1
        self.evaluation_workers.foreach_worker_async(func=remote_fn, healthy_only=True)
        eval_results = self.evaluation_workers.fetch_ready_async_reqs()
        batches = []
        i = 0
        for _, result in eval_results:
            batch, metrics, seq_no = result
            if seq_no == self._evaluation_weights_seq_number and i * (1 if unit == 'episodes' else rollout * num_envs) < units_left_to_do:
                batches.append(batch)
                rollout_metrics.extend(metrics)
            i += 1
        _agent_steps = sum((b.agent_steps() for b in batches))
        _env_steps = sum((b.env_steps() for b in batches))
        if unit == 'episodes':
            num_units_done += len(batches)
            for ma_batch in batches:
                ma_batch = ma_batch.as_multi_agent()
                for batch in ma_batch.policy_batches.values():
                    assert batch.is_terminated_or_truncated()
        else:
            num_units_done += _agent_steps if self.config.count_steps_by == 'agent_steps' else _env_steps
        if self.reward_estimators:
            all_batches.extend(batches)
        agent_steps_this_iter += _agent_steps
        env_steps_this_iter += _env_steps
        logger.info(f'Ran round {_round} of parallel evaluation ({num_units_done}/{(duration if not auto else '?')} {unit} done)')
    sampler_results = summarize_episodes(rollout_metrics, keep_custom_metrics=eval_cfg['keep_per_episode_custom_metrics'])
    metrics = dict({'sampler_results': sampler_results}, **sampler_results)
    metrics[NUM_AGENT_STEPS_SAMPLED_THIS_ITER] = agent_steps_this_iter
    metrics[NUM_ENV_STEPS_SAMPLED_THIS_ITER] = env_steps_this_iter
    metrics['timesteps_this_iter'] = env_steps_this_iter
    if self.reward_estimators:
        metrics['off_policy_estimator'] = {}
        total_batch = concat_samples(all_batches)
        for name, estimator in self.reward_estimators.items():
            estimates = estimator.estimate(total_batch)
            metrics['off_policy_estimator'][name] = estimates
    self.evaluation_metrics = {'evaluation': metrics}
    self.callbacks.on_evaluate_end(algorithm=self, evaluation_metrics=self.evaluation_metrics)
    return self.evaluation_metrics
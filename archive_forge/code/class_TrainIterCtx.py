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
class TrainIterCtx:

    def __init__(self, algo: Algorithm):
        self.algo = algo
        self.time_start = None
        self.time_stop = None

    def __enter__(self):
        self.failures = -1
        self.time_start = time.time()
        self.sampled = 0
        self.trained = 0
        self.init_env_steps_sampled = self.algo._counters[NUM_ENV_STEPS_SAMPLED]
        self.init_env_steps_trained = self.algo._counters[NUM_ENV_STEPS_TRAINED]
        self.init_agent_steps_sampled = self.algo._counters[NUM_AGENT_STEPS_SAMPLED]
        self.init_agent_steps_trained = self.algo._counters[NUM_AGENT_STEPS_TRAINED]
        self.failure_tolerance = self.algo.config['num_consecutive_worker_failures_tolerance']
        return self

    def __exit__(self, *args):
        self.time_stop = time.time()

    def get_time_taken_sec(self) -> float:
        """Returns the time we spent in the context in seconds."""
        return self.time_stop - self.time_start

    def should_stop(self, results):
        if results is None:
            self.failures += 1
            if self.failures > self.failure_tolerance:
                raise RuntimeError(f'More than `num_consecutive_worker_failures_tolerance={self.failure_tolerance}` consecutive worker failures! Exiting.')
            return False
        elif self.algo.config._disable_execution_plan_api:
            if self.algo.config.count_steps_by == 'agent_steps':
                self.sampled = self.algo._counters[NUM_AGENT_STEPS_SAMPLED] - self.init_agent_steps_sampled
                self.trained = self.algo._counters[NUM_AGENT_STEPS_TRAINED] - self.init_agent_steps_trained
            else:
                self.sampled = self.algo._counters[NUM_ENV_STEPS_SAMPLED] - self.init_env_steps_sampled
                self.trained = self.algo._counters[NUM_ENV_STEPS_TRAINED] - self.init_env_steps_trained
            min_t = self.algo.config['min_time_s_per_iteration']
            min_sample_ts = self.algo.config['min_sample_timesteps_per_iteration']
            min_train_ts = self.algo.config['min_train_timesteps_per_iteration']
            if (not min_t or time.time() - self.time_start >= min_t) and (not min_sample_ts or self.sampled >= min_sample_ts) and (not min_train_ts or self.trained >= min_train_ts):
                return True
            else:
                return False
        else:
            return True
import copy
import logging
import math
import os
import sys
from typing import (
from packaging import version
import ray
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.core.learner.learner import LearnerHyperparameters
from ray.rllib.core.learner.learner_group_config import LearnerGroupConfig, ModuleSpec
from ray.rllib.core.rl_module.marl_module import MultiAgentRLModuleSpec
from ray.rllib.core.rl_module.rl_module import ModuleID, SingleAgentRLModuleSpec
from ray.rllib.core.learner.learner import TorchCompileWhatToCompile
from ray.rllib.env.env_context import EnvContext
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.env.wrappers.atari_wrappers import is_atari
from ray.rllib.evaluation.collectors.sample_collector import SampleCollector
from ray.rllib.evaluation.collectors.simple_list_collector import SimpleListCollector
from ray.rllib.evaluation.rollout_worker import RolloutWorker
from ray.rllib.models import MODEL_DEFAULTS
from ray.rllib.policy.policy import Policy, PolicySpec
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.utils import deep_update, merge_dicts
from ray.rllib.utils.annotations import (
from ray.rllib.utils.deprecation import (
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.from_config import NotProvided, from_config
from ray.rllib.utils.gym import (
from ray.rllib.utils.policy import validate_policy_id
from ray.rllib.utils.schedules.scheduler import Scheduler
from ray.rllib.utils.serialization import (
from ray.rllib.utils.torch_utils import TORCH_COMPILE_REQUIRED_VERSION
from ray.rllib.utils.typing import (
from ray.tune.logger import Logger
from ray.tune.registry import get_trainable_cls
from ray.tune.result import TRIAL_INFO
from ray.tune.tune import _Config
@staticmethod
def _translate_special_keys(key: str, warn_deprecated: bool=True) -> str:
    if key == 'callbacks':
        key = 'callbacks_class'
    elif key == 'create_env_on_driver':
        key = 'create_env_on_local_worker'
    elif key == 'custom_eval_function':
        key = 'custom_evaluation_function'
    elif key == 'framework':
        key = 'framework_str'
    elif key == 'input':
        key = 'input_'
    elif key == 'lambda':
        key = 'lambda_'
    elif key == 'num_cpus_for_driver':
        key = 'num_cpus_for_local_worker'
    elif key == 'num_workers':
        key = 'num_rollout_workers'
    if warn_deprecated:
        if key == 'collect_metrics_timeout':
            deprecation_warning(old='collect_metrics_timeout', new='metrics_episode_collection_timeout_s', error=True)
        elif key == 'metrics_smoothing_episodes':
            deprecation_warning(old='config.metrics_smoothing_episodes', new='config.metrics_num_episodes_for_smoothing', error=True)
        elif key == 'min_iter_time_s':
            deprecation_warning(old='config.min_iter_time_s', new='config.min_time_s_per_iteration', error=True)
        elif key == 'min_time_s_per_reporting':
            deprecation_warning(old='config.min_time_s_per_reporting', new='config.min_time_s_per_iteration', error=True)
        elif key == 'min_sample_timesteps_per_reporting':
            deprecation_warning(old='config.min_sample_timesteps_per_reporting', new='config.min_sample_timesteps_per_iteration', error=True)
        elif key == 'min_train_timesteps_per_reporting':
            deprecation_warning(old='config.min_train_timesteps_per_reporting', new='config.min_train_timesteps_per_iteration', error=True)
        elif key == 'timesteps_per_iteration':
            deprecation_warning(old='config.timesteps_per_iteration', new='`config.min_sample_timesteps_per_iteration` OR `config.min_train_timesteps_per_iteration`', error=True)
        elif key == 'evaluation_num_episodes':
            deprecation_warning(old='config.evaluation_num_episodes', new='`config.evaluation_duration` and `config.evaluation_duration_unit=episodes`', error=True)
    return key
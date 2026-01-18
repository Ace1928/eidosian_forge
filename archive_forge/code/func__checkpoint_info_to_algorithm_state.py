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
@staticmethod
def _checkpoint_info_to_algorithm_state(checkpoint_info: dict, policy_ids: Optional[Container[PolicyID]]=None, policy_mapping_fn: Optional[Callable[[AgentID, EpisodeID], PolicyID]]=None, policies_to_train: Optional[Union[Container[PolicyID], Callable[[PolicyID, Optional[SampleBatchType]], bool]]]=None) -> Dict:
    """Converts a checkpoint info or object to a proper Algorithm state dict.

        The returned state dict can be used inside self.__setstate__().

        Args:
            checkpoint_info: A checkpoint info dict as returned by
                `ray.rllib.utils.checkpoints.get_checkpoint_info(
                [checkpoint dir or AIR Checkpoint])`.
            policy_ids: Optional list/set of PolicyIDs. If not None, only those policies
                listed here will be included in the returned state. Note that
                state items such as filters, the `is_policy_to_train` function, as
                well as the multi-agent `policy_ids` dict will be adjusted as well,
                based on this arg.
            policy_mapping_fn: An optional (updated) policy mapping function
                to include in the returned state.
            policies_to_train: An optional list of policy IDs to be trained
                or a callable taking PolicyID and SampleBatchType and
                returning a bool (trainable or not?) to include in the returned state.

        Returns:
             The state dict usable within the `self.__setstate__()` method.
        """
    if checkpoint_info['type'] != 'Algorithm':
        raise ValueError(f'`checkpoint` arg passed to `Algorithm._checkpoint_info_to_algorithm_state()` must be an Algorithm checkpoint (but is {checkpoint_info['type']})!')
    msgpack = None
    if checkpoint_info.get('format') == 'msgpack':
        msgpack = try_import_msgpack(error=True)
    with open(checkpoint_info['state_file'], 'rb') as f:
        if msgpack is not None:
            state = msgpack.load(f)
        else:
            state = pickle.load(f)
    if checkpoint_info['checkpoint_version'] > version.Version('0.1') and state.get('worker') is not None:
        worker_state = state['worker']
        policy_ids = set(policy_ids if policy_ids is not None else worker_state['policy_ids'])
        worker_state['filters'] = {pid: filter for pid, filter in worker_state['filters'].items() if pid in policy_ids}
        if isinstance(state['algorithm_class'], str):
            state['algorithm_class'] = deserialize_type(state['algorithm_class']) or get_trainable_cls(state['algorithm_class'])
        default_config = state['algorithm_class'].get_default_config()
        if isinstance(default_config, AlgorithmConfig):
            new_config = default_config.update_from_dict(state['config'])
        else:
            new_config = Algorithm.merge_algorithm_configs(default_config, state['config'])
        new_policies = new_config.policies
        if isinstance(new_policies, (set, list, tuple)):
            new_policies = {pid for pid in new_policies if pid in policy_ids}
        else:
            new_policies = {pid: spec for pid, spec in new_policies.items() if pid in policy_ids}
        new_config.multi_agent(policies=new_policies, policies_to_train=policies_to_train, **{'policy_mapping_fn': policy_mapping_fn} if policy_mapping_fn is not None else {})
        state['config'] = new_config
        worker_state['policy_states'] = {}
        for pid in policy_ids:
            policy_state_file = os.path.join(checkpoint_info['checkpoint_dir'], 'policies', pid, 'policy_state.' + ('msgpck' if checkpoint_info['format'] == 'msgpack' else 'pkl'))
            if not os.path.isfile(policy_state_file):
                raise ValueError(f'Given checkpoint does not seem to be valid! No policy state file found for PID={pid}. The file not found is: {policy_state_file}.')
            with open(policy_state_file, 'rb') as f:
                if msgpack is not None:
                    worker_state['policy_states'][pid] = msgpack.load(f)
                else:
                    worker_state['policy_states'][pid] = pickle.load(f)
        if policy_mapping_fn is not None:
            worker_state['policy_mapping_fn'] = policy_mapping_fn
        if policies_to_train is not None or worker_state['is_policy_to_train'] == NOT_SERIALIZABLE:
            worker_state['is_policy_to_train'] = policies_to_train
    return state
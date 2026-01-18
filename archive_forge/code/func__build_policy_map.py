import copy
import importlib.util
import logging
import os
import platform
import threading
from collections import defaultdict
from types import FunctionType
from typing import (
import numpy as np
import tree  # pip install dm_tree
from gymnasium.spaces import Discrete, MultiDiscrete, Space
import ray
from ray import ObjectRef
from ray import cloudpickle as pickle
from ray.rllib.connectors.util import (
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.env.base_env import BaseEnv, convert_to_base_env
from ray.rllib.env.env_context import EnvContext
from ray.rllib.env.env_runner import EnvRunner
from ray.rllib.env.external_multi_agent_env import ExternalMultiAgentEnv
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.env.wrappers.atari_wrappers import is_atari, wrap_deepmind
from ray.rllib.evaluation.metrics import RolloutMetrics
from ray.rllib.evaluation.sampler import SyncSampler
from ray.rllib.models import ModelCatalog
from ray.rllib.models.preprocessors import Preprocessor
from ray.rllib.offline import (
from ray.rllib.policy.policy import Policy, PolicySpec
from ray.rllib.policy.policy_map import PolicyMap
from ray.rllib.policy.sample_batch import (
from ray.rllib.policy.torch_policy import TorchPolicy
from ray.rllib.policy.torch_policy_v2 import TorchPolicyV2
from ray.rllib.utils import check_env, force_list
from ray.rllib.utils.annotations import DeveloperAPI, override
from ray.rllib.utils.debug import summarize, update_global_seed_if_necessary
from ray.rllib.utils.deprecation import DEPRECATED_VALUE, deprecation_warning
from ray.rllib.utils.error import ERR_MSG_NO_GPUS, HOWTO_CHANGE_CONFIG
from ray.rllib.utils.filter import Filter, NoFilter, get_filter
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.from_config import from_config
from ray.rllib.utils.policy import create_policy_for_framework, validate_policy_id
from ray.rllib.utils.sgd import do_minibatch_sgd
from ray.rllib.utils.tf_run_builder import _TFRunBuilder
from ray.rllib.utils.tf_utils import get_gpu_devices as get_tf_gpu_devices
from ray.rllib.utils.tf_utils import get_tf_eager_cls_if_necessary
from ray.rllib.utils.typing import (
from ray.tune.registry import registry_contains_input, registry_get_input
from ray.util.annotations import PublicAPI
from ray.util.debug import disable_log_once_globally, enable_periodic_logging, log_once
from ray.util.iter import ParallelIteratorWorker
def _build_policy_map(self, *, policy_dict: MultiAgentPolicyConfigDict, policy: Optional[Policy]=None, policy_states: Optional[Dict[PolicyID, PolicyState]]=None) -> None:
    """Adds the given policy_dict to `self.policy_map`.

        Args:
            policy_dict: The MultiAgentPolicyConfigDict to be added to this
                worker's PolicyMap.
            policy: If the policy to add already exists, user can provide it here.
            policy_states: Optional dict from PolicyIDs to PolicyStates to
                restore the states of the policies being built.
        """
    self.policy_map = self.policy_map or PolicyMap(capacity=self.config.policy_map_capacity, policy_states_are_swappable=self.config.policy_states_are_swappable)
    for name, policy_spec in sorted(policy_dict.items()):
        if policy is None:
            new_policy = create_policy_for_framework(policy_id=name, policy_class=get_tf_eager_cls_if_necessary(policy_spec.policy_class, policy_spec.config), merged_config=policy_spec.config, observation_space=policy_spec.observation_space, action_space=policy_spec.action_space, worker_index=self.worker_index, seed=self.seed)
        else:
            new_policy = policy
        if self.config.get('_enable_new_api_stack', False) and self.config.get('torch_compile_worker'):
            if self.config.framework_str != 'torch':
                raise ValueError('Attempting to compile a non-torch RLModule.')
            rl_module = getattr(new_policy, 'model', None)
            if rl_module is not None:
                compile_config = self.config.get_torch_compile_worker_config()
                rl_module.compile(compile_config)
        self.policy_map[name] = new_policy
        restore_states = (policy_states or {}).get(name, None)
        if restore_states:
            new_policy.set_state(restore_states)
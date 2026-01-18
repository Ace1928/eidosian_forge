import json
import logging
import os
import platform
from abc import ABCMeta, abstractmethod
from typing import (
import gymnasium as gym
import numpy as np
import tree  # pip install dm_tree
from gymnasium.spaces import Box
from packaging import version
import ray
import ray.cloudpickle as pickle
from ray.actor import ActorHandle
from ray.train import Checkpoint
from ray.rllib.core.models.base import STATE_IN, STATE_OUT
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils.annotations import (
from ray.rllib.utils.checkpoints import (
from ray.rllib.utils.deprecation import (
from ray.rllib.utils.exploration.exploration import Exploration
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.from_config import from_config
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.serialization import (
from ray.rllib.utils.spaces.space_utils import (
from ray.rllib.utils.tensor_dtype import get_np_dtype
from ray.rllib.utils.tf_utils import get_tf_eager_cls_if_necessary
from ray.rllib.utils.typing import (
from ray.util.annotations import PublicAPI
def _get_num_gpus_for_policy(self) -> int:
    """Decide on the number of CPU/GPU nodes this policy should run on.

        Return:
            0 if policy should run on CPU. >0 if policy should run on 1 or
            more GPUs.
        """
    worker_idx = self.config.get('worker_index', 0)
    fake_gpus = self.config.get('_fake_gpus', False)
    if ray._private.worker._mode() == ray._private.worker.LOCAL_MODE and (not fake_gpus):
        num_gpus = 0
    elif worker_idx == 0:
        if self.config.get('_enable_new_api_stack', False):
            num_gpus = self.config['num_gpus_per_worker']
        else:
            num_gpus = self.config['num_gpus']
    else:
        num_gpus = self.config['num_gpus_per_worker']
    if num_gpus == 0:
        dev = 'CPU'
    else:
        dev = '{} {}'.format(num_gpus, 'fake-GPUs' if fake_gpus else 'GPUs')
    logger.info('Policy (worker={}) running on {}.'.format(worker_idx if worker_idx > 0 else 'local', dev))
    return num_gpus
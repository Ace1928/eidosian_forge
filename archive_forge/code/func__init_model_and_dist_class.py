import copy
import functools
import logging
import math
import os
import threading
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, Type, Union
import gymnasium as gym
import numpy as np
import tree  # pip install dm_tree
import ray
from ray.rllib.core.models.base import STATE_IN, STATE_OUT
from ray.rllib.core.rl_module import RLModule
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.rnn_sequencing import pad_batch_to_sequences_of_same_size
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_policy import _directStepOptimizerSingleton
from ray.rllib.utils import NullContextManager, force_list
from ray.rllib.utils.annotations import (
from ray.rllib.utils.annotations import ExperimentalAPI
from ray.rllib.utils.error import ERR_MSG_TORCH_POLICY_CANNOT_SAVE_MODEL
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.metrics import (
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.spaces.space_utils import normalize_action
from ray.rllib.utils.threading import with_lock
from ray.rllib.utils.torch_utils import convert_to_torch_tensor
from ray.rllib.utils.typing import (
def _init_model_and_dist_class(self):
    if is_overridden(self.make_model) and is_overridden(self.make_model_and_action_dist):
        raise ValueError('Only one of make_model or make_model_and_action_dist can be overridden.')
    if is_overridden(self.make_model):
        model = self.make_model()
        dist_class, _ = ModelCatalog.get_action_dist(self.action_space, self.config['model'], framework=self.framework)
    elif is_overridden(self.make_model_and_action_dist):
        model, dist_class = self.make_model_and_action_dist()
    else:
        dist_class, logit_dim = ModelCatalog.get_action_dist(self.action_space, self.config['model'], framework=self.framework)
        model = ModelCatalog.get_model_v2(obs_space=self.observation_space, action_space=self.action_space, num_outputs=logit_dim, model_config=self.config['model'], framework=self.framework)
    return (model, dist_class)
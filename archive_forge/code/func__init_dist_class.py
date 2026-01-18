from collections import OrderedDict
import gymnasium as gym
import logging
import re
import tree  # pip install dm_tree
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Type, Union
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.tf_action_dist import TFActionDistribution
from ray.rllib.policy.dynamic_tf_policy import TFMultiGPUTowerStack
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_policy import TFPolicy
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils import force_list
from ray.rllib.utils.annotations import (
from ray.rllib.utils.debug import summarize
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.metrics import (
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY
from ray.rllib.utils.spaces.space_utils import get_dummy_batch_for_space
from ray.rllib.utils.tf_utils import get_placeholder
from ray.rllib.utils.typing import (
from ray.util.debug import log_once
def _init_dist_class(self):
    if is_overridden(self.action_sampler_fn) or is_overridden(self.action_distribution_fn):
        if not is_overridden(self.make_model):
            raise ValueError('`make_model` is required if `action_sampler_fn` OR `action_distribution_fn` is given')
        return None
    else:
        dist_class, _ = ModelCatalog.get_action_dist(self.action_space, self.config['model'])
        return dist_class
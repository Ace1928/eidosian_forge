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
def _do_loss_init(self, train_batch: SampleBatch):
    losses = self.loss(self.model, self.dist_class, train_batch)
    losses = force_list(losses)
    self._stats_fetches.update(self.stats_fn(train_batch))
    self._update_ops = []
    if not isinstance(self.model, tf.keras.Model):
        self._update_ops = self.model.update_ops()
    return losses
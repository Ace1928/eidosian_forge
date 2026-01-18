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
def _init_state_inputs(self, existing_inputs: Dict[str, 'tf1.placeholder']):
    """Initialize input placeholders.

        Args:
            existing_inputs: existing placeholders.
        """
    if existing_inputs:
        self._state_inputs = [v for k, v in existing_inputs.items() if k.startswith('state_in_')]
        if self._state_inputs:
            self._seq_lens = existing_inputs[SampleBatch.SEQ_LENS]
    else:
        self._state_inputs = [get_placeholder(space=vr.space, time_axis=not isinstance(vr.shift, int), name=k) for k, vr in self.model.view_requirements.items() if k.startswith('state_in_')]
        if self._state_inputs:
            self._seq_lens = tf1.placeholder(dtype=tf.int32, shape=[None], name='seq_lens')
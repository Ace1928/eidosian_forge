from collections import namedtuple, OrderedDict
import gymnasium as gym
import logging
import re
import tree  # pip install dm_tree
from typing import Callable, Dict, List, Optional, Tuple, Type, Union
from ray.util.debug import log_once
from ray.rllib.models.tf.tf_action_dist import TFActionDistribution
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_policy import TFPolicy
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.utils import force_list
from ray.rllib.utils.annotations import override, DeveloperAPI
from ray.rllib.utils.debug import summarize
from ray.rllib.utils.deprecation import (
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.metrics import (
from ray.rllib.utils.spaces.space_utils import get_dummy_batch_for_space
from ray.rllib.utils.tf_utils import get_placeholder
from ray.rllib.utils.typing import (
def _setup_device(self, tower_i, device, device_input_placeholders, num_data_in):
    assert num_data_in <= len(device_input_placeholders)
    with tf.device(device):
        with tf1.name_scope(TOWER_SCOPE_NAME + f'_{tower_i}'):
            device_input_batches = []
            device_input_slices = []
            for i, ph in enumerate(device_input_placeholders):
                current_batch = tf1.Variable(ph, trainable=False, validate_shape=False, collections=[])
                device_input_batches.append(current_batch)
                if i < num_data_in:
                    scale = self._max_seq_len
                    granularity = self._max_seq_len
                else:
                    scale = self._max_seq_len
                    granularity = 1
                current_slice = tf.slice(current_batch, [self._batch_index // scale * granularity] + [0] * len(ph.shape[1:]), [self._per_device_batch_size // scale * granularity] + [-1] * len(ph.shape[1:]))
                current_slice.set_shape(ph.shape)
                device_input_slices.append(current_slice)
            graph_obj = self.policy_copy(device_input_slices)
            device_grads = graph_obj.gradients(self.optimizers, graph_obj._losses)
        return _Tower(tf.group(*[batch.initializer for batch in device_input_batches]), device_grads, graph_obj)
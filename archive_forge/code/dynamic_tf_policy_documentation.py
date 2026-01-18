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
Run a single step of SGD.

        Runs a SGD step over a slice of the preloaded batch with size given by
        self._loaded_per_device_batch_size and offset given by the batch_index
        argument.

        Updates shared model weights based on the averaged per-device
        gradients.

        Args:
            sess: TensorFlow session.
            batch_index: Offset into the preloaded data. This value must be
                between `0` and `tuples_per_device`. The amount of data to
                process is at most `max_per_device_batch_size`.

        Returns:
            The outputs of extra_ops evaluated over the batch.
        
import functools
import logging
import os
import threading
from typing import Dict, List, Optional, Tuple, Union
import tree  # pip install dm_tree
from ray.rllib.evaluation.episode import Episode
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.repeated_values import RepeatedValues
from ray.rllib.policy.policy import Policy, PolicyState
from ray.rllib.policy.rnn_sequencing import pad_batch_to_sequences_of_same_size
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils import add_mixins, force_list
from ray.rllib.utils.annotations import override, DeveloperAPI
from ray.rllib.utils.deprecation import (
from ray.rllib.utils.error import ERR_MSG_TF_POLICY_CANNOT_SAVE_KERAS_MODEL
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.metrics import (
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.spaces.space_utils import normalize_action
from ray.rllib.utils.tf_utils import get_gpu_devices
from ray.rllib.utils.threading import with_lock
from ray.rllib.utils.typing import (
from ray.util.debug import log_once
def _convert_to_tf(x, dtype=None):
    if isinstance(x, SampleBatch):
        dict_ = {k: v for k, v in x.items() if k != SampleBatch.INFOS}
        return tree.map_structure(_convert_to_tf, dict_)
    elif isinstance(x, Policy):
        return x
    elif isinstance(x, RepeatedValues):
        return RepeatedValues(tree.map_structure(_convert_to_tf, x.values), x.lengths, x.max_len)
    if x is not None:
        d = dtype
        return tree.map_structure(lambda f: _convert_to_tf(f, d) if isinstance(f, RepeatedValues) else tf.convert_to_tensor(f, d) if f is not None and (not tf.is_tensor(f)) else f, x)
    return x
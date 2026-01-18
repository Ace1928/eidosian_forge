import logging
import queue
import time
from abc import ABCMeta, abstractmethod
from collections import defaultdict, namedtuple
from typing import (
import numpy as np
import tree  # pip install dm_tree
from ray.rllib.env.base_env import ASYNC_RESET_RETURN, BaseEnv, convert_to_base_env
from ray.rllib.evaluation.collectors.sample_collector import SampleCollector
from ray.rllib.evaluation.collectors.simple_list_collector import SimpleListCollector
from ray.rllib.evaluation.env_runner_v2 import (
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.metrics import RolloutMetrics
from ray.rllib.offline import InputReader
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.policy_map import PolicyMap
from ray.rllib.policy.sample_batch import SampleBatch, concat_samples
from ray.rllib.utils.annotations import DeveloperAPI, override
from ray.rllib.utils.debug import summarize
from ray.rllib.utils.deprecation import deprecation_warning, DEPRECATED_VALUE
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.numpy import convert_to_numpy, make_action_immutable
from ray.rllib.utils.spaces.space_utils import clip_action, unbatch, unsquash_action
from ray.rllib.utils.typing import (
from ray.util.debug import log_once
def _to_column_format(rnn_state_rows: List[List[Any]]) -> StateBatch:
    num_cols = len(rnn_state_rows[0])
    return [[row[i] for row in rnn_state_rows] for i in range(num_cols)]
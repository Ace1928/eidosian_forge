from collections import defaultdict
import logging
import time
import tree  # pip install dm_tree
from typing import TYPE_CHECKING, Dict, Iterator, List, Optional, Set, Tuple, Union
import numpy as np
from ray.rllib.env.base_env import ASYNC_RESET_RETURN, BaseEnv
from ray.rllib.env.external_env import ExternalEnvWrapper
from ray.rllib.env.wrappers.atari_wrappers import MonitorEnv, get_wrapper_by_cls
from ray.rllib.evaluation.collectors.simple_list_collector import _PolicyCollectorGroup
from ray.rllib.policy.rnn_sequencing import pad_batch_to_sequences_of_same_size
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.evaluation.metrics import RolloutMetrics
from ray.rllib.models.preprocessors import Preprocessor
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import MultiAgentBatch, SampleBatch, concat_samples
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.utils.filter import Filter
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.spaces.space_utils import unbatch, get_original_space
from ray.rllib.utils.typing import (
from ray.util.debug import log_once
def _get_avg(self):
    factor = MS_TO_SEC / self.iters
    return {'mean_raw_obs_processing_ms': self.raw_obs_processing_time * factor, 'mean_inference_ms': self.inference_time * factor, 'mean_action_processing_ms': self.action_processing_time * factor, 'mean_env_wait_ms': self.env_wait_time * factor, 'mean_env_render_ms': self.env_render_time * factor}
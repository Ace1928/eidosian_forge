import collections
import logging
from enum import Enum
from typing import Any, Dict, Optional
from ray.util.timer import _Timer
from ray.rllib.policy.rnn_sequencing import timeslice_along_seq_lens_with_overlap
from ray.rllib.policy.sample_batch import MultiAgentBatch, SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.deprecation import Deprecated
from ray.rllib.utils.from_config import from_config
from ray.rllib.utils.replay_buffers.replay_buffer import (
from ray.rllib.utils.typing import PolicyID, SampleBatchType
from ray.util.annotations import DeveloperAPI
from ray.util.debug import log_once
@DeveloperAPI
class ReplayMode(Enum):
    LOCKSTEP = 'lockstep'
    INDEPENDENT = 'independent'
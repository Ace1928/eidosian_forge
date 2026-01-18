from enum import Enum
import logging
import numpy as np
import random
from typing import Any, Dict, List, Optional, Union
import ray  # noqa F401
import psutil
from ray.rllib.policy.sample_batch import SampleBatch, concat_samples
from ray.rllib.utils.actor_manager import FaultAwareApply
from ray.rllib.utils.annotations import override
from ray.rllib.utils.deprecation import Deprecated
from ray.rllib.utils.metrics.window_stat import WindowStat
from ray.rllib.utils.replay_buffers.base import ReplayBufferInterface
from ray.rllib.utils.typing import SampleBatchType
from ray.util.annotations import DeveloperAPI
from ray.util.debug import log_once
@DeveloperAPI
class StorageUnit(Enum):
    """Specifies how batches are structured in a ReplayBuffer.

    timesteps: One buffer slot per timestep.
    sequences: One buffer slot per sequence.
    episodes: One buffer slot per episode.
    fragemts: One buffer slot per incoming batch.
    """
    TIMESTEPS = 'timesteps'
    SEQUENCES = 'sequences'
    EPISODES = 'episodes'
    FRAGMENTS = 'fragments'
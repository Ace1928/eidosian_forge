import gc
import os
import platform
import tracemalloc
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Type, Union
import numpy as np
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.env.env_context import EnvContext
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import (
from ray.rllib.utils.deprecation import Deprecated, deprecation_warning
from ray.rllib.utils.exploration.random_encoder import (
from ray.rllib.utils.typing import AgentID, EnvType, PolicyID
from ray.tune.callback import _CallbackMeta
import psutil
class MemoryTrackingCallbacks(DefaultCallbacks):
    """MemoryTrackingCallbacks can be used to trace and track memory usage
    in rollout workers.

    The Memory Tracking Callbacks uses tracemalloc and psutil to track
    python allocations during rollouts,
    in training or evaluation.

    The tracking data is logged to the custom_metrics of an episode and
    can therefore be viewed in tensorboard
    (or in WandB etc..)

    Add MemoryTrackingCallbacks callback to the tune config
    e.g. { ...'callbacks': MemoryTrackingCallbacks ...}

    Note:
        This class is meant for debugging and should not be used
        in production code as tracemalloc incurs
        a significant slowdown in execution speed.
    """

    def __init__(self):
        super().__init__()
        tracemalloc.start(10)

    @override(DefaultCallbacks)
    def on_episode_end(self, *, worker: 'RolloutWorker', base_env: BaseEnv, policies: Dict[PolicyID, Policy], episode: Union[Episode, EpisodeV2, Exception], env_index: Optional[int]=None, **kwargs) -> None:
        gc.collect()
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        for stat in top_stats[:10]:
            count = stat.count
            size = stat.size / 1024
            trace = str(stat.traceback)
            episode.custom_metrics[f'tracemalloc/{trace}/size'] = size
            episode.custom_metrics[f'tracemalloc/{trace}/count'] = count
        process = psutil.Process(os.getpid())
        worker_rss = process.memory_info().rss
        worker_vms = process.memory_info().vms
        if platform.system() == 'Linux':
            worker_data = process.memory_info().data
            episode.custom_metrics['tracemalloc/worker/data'] = worker_data
        episode.custom_metrics['tracemalloc/worker/rss'] = worker_rss
        episode.custom_metrics['tracemalloc/worker/vms'] = worker_vms
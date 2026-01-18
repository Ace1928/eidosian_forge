from collections import defaultdict, deque
from functools import partial
import pathlib
from typing import (
import uuid
import ray
from ray.rllib.core.learner.reduce_result_dict_fn import _reduce_mean_results
from ray.rllib.core.rl_module.rl_module import (
from ray.rllib.core.learner.learner import LearnerSpec
from ray.rllib.policy.sample_batch import MultiAgentBatch
from ray.rllib.utils.actor_manager import FaultTolerantActorManager
from ray.rllib.utils.minibatch_utils import ShardBatchIterator
from ray.rllib.utils.typing import ResultDict
from ray.rllib.utils.numpy import convert_to_numpy
from ray.train._internal.backend_executor import BackendExecutor
from ray.tune.utils.file_transfer import sync_dir_between_nodes
from ray.util.annotations import PublicAPI
@staticmethod
def _resolve_checkpoint_path(path: str) -> pathlib.Path:
    """Checks that the provided checkpoint path is a dir and makes it absolute."""
    path = pathlib.Path(path)
    if not path.is_dir():
        raise ValueError(f'Path {path} is not a directory. Please specify a directory containing the checkpoint files.')
    if not path.exists():
        raise ValueError(f'Path {path} does not exist.')
    path = path.absolute()
    return path
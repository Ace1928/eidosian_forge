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
def _get_ip_address(_=None) -> str:
    """Returns this process's address.

        Args:
            _: Unused arg. Exists to make this function compatible with foreach_actor
            calls.

        Returns:
            The address of this process.

        """
    import ray
    return ray.util.get_node_ip_address()
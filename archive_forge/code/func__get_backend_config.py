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
def _get_backend_config(learner_class: Type['Learner']) -> str:
    if learner_class.framework == 'torch':
        from ray.train.torch import TorchConfig
        backend_config = TorchConfig()
    elif learner_class.framework == 'tf2':
        from ray.train.tensorflow import TensorflowConfig
        backend_config = TensorflowConfig()
    else:
        raise ValueError('framework must be either torch or tf')
    return backend_config
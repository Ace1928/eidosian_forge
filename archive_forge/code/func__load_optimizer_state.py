import json
import logging
import pathlib
from typing import (
from ray.rllib.core.learner.learner import (
from ray.rllib.core.rl_module.rl_module import (
from ray.rllib.core.rl_module.tf.tf_rl_module import TfRLModule
from ray.rllib.policy.eager_tf_policy import _convert_to_tf
from ray.rllib.policy.sample_batch import MultiAgentBatch
from ray.rllib.utils.annotations import (
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.metrics import ALL_MODULES
from ray.rllib.utils.nested_dict import NestedDict
from ray.rllib.utils.serialization import convert_numpy_to_python_primitives
from ray.rllib.utils.typing import Optimizer, Param, ParamDict, TensorType
def _load_optimizer_state(self, path: pathlib.Path, optim: 'tf.keras.optimizers.Optimizer', optim_name: str) -> None:
    """Load the state of optim from the state saved at path/optim_name_state.txt.

        Args:
            path: The path to the directory to load the state from.
            optim: The optimizer to load the state into.
            optim_name: The name of the optimizer.

        """
    contents = tf.io.read_file(str(path / f'{optim_name}_state.txt'))
    serialized_tensors = tf.strings.split(contents, sep='tensor: ')
    unserialized_optim_state = []
    for serialized_tensor, optim_tensor in zip(serialized_tensors, optim.variables()):
        unserialized_optim_state.append(tf.io.parse_tensor(serialized_tensor, optim_tensor.dtype))
    optim.set_weights(unserialized_optim_state)
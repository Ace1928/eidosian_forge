import abc
import json
import logging
import pathlib
from collections import defaultdict
from enum import Enum
from dataclasses import dataclass, field
from typing import (
import ray
from ray.rllib.core.learner.reduce_result_dict_fn import _reduce_mean_results
from ray.rllib.core.learner.scaling_config import LearnerGroupScalingConfig
from ray.rllib.core.rl_module.marl_module import (
from ray.rllib.core.rl_module.rl_module import (
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID, MultiAgentBatch
from ray.rllib.utils.annotations import (
from ray.rllib.utils.debug import update_global_seed_if_necessary
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.metrics import (
from ray.rllib.utils.minibatch_utils import (
from ray.rllib.utils.nested_dict import NestedDict
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.schedules.scheduler import Scheduler
from ray.rllib.utils.serialization import serialize_type
from ray.rllib.utils.typing import (
from ray.util.annotations import PublicAPI
def _check_result(self, result: Mapping[str, Any]) -> None:
    """Checks whether the result has the correct format.

        All the keys should be referencing the module ids that got updated. There is a
        special key `ALL_MODULES` that hold any extra information that is not specific
        to a module.

        Args:
            result: The result of the update.

        Raises:
            ValueError: If the result are not in the correct format.
        """
    if not isinstance(result, dict):
        raise ValueError(f'The result of the update must be a dictionary. Got: {type(result)}')
    if ALL_MODULES not in result:
        raise ValueError(f'The result of the update must have a key {ALL_MODULES} that holds any extra information that is not specific to a module.')
    for key in result:
        if key != ALL_MODULES:
            if key not in self.module.keys():
                raise ValueError(f'The key {key} in the result of the update is not a valid module id. Valid module ids are: {list(self.module.keys())}.')
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
@dataclass
class LearnerHyperparameters:
    """Hyperparameters for a Learner, derived from a subset of AlgorithmConfig values.

    Instances of this class should only be created via calling
    `get_learner_hyperparameters()` on a frozen AlgorithmConfig object and should always
    considered read-only.

    When creating a new Learner, you should also define a new sub-class of this class
    and make sure the respective AlgorithmConfig sub-class has a proper implementation
    of the `get_learner_hyperparameters` method.

    Validation of the values of these hyperparameters should be done by the
    respective AlgorithmConfig class.

    For configuring different learning behaviors for different (single-agent) RLModules
    within the Learner, RLlib uses the `_per_module_overrides` property (dict), mapping
    ModuleID to a overridden version of self, in which the module-specific override
    settings are applied.
    """
    learning_rate: LearningRateOrSchedule = None
    grad_clip: float = None
    grad_clip_by: str = None
    seed: int = None
    _per_module_overrides: Optional[Dict[ModuleID, 'LearnerHyperparameters']] = None

    def get_hps_for_module(self, module_id: ModuleID) -> 'LearnerHyperparameters':
        """Returns a LearnerHyperparameter instance, given a `module_id`.

        This is useful for passing these module-specific HPs to a Learner's
        `..._for_module(module_id=.., hps=..)` methods. Individual modules within
        a MultiAgentRLModule can then override certain AlgorithmConfig settings
        of the main config, e.g. the learning rate.

        Args:
            module_id: The module ID for which to return a specific
                LearnerHyperparameter instance.

        Returns:
            The module specific LearnerHyperparameter instance.
        """
        if self._per_module_overrides is not None and module_id in self._per_module_overrides:
            if isinstance(self._per_module_overrides[module_id], dict):
                self._per_module_overrides[module_id] = type(self)(**self._per_module_overrides[module_id])
            return self._per_module_overrides[module_id]
        else:
            return self
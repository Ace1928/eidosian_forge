import abc
from dataclasses import dataclass
from typing import Any, Mapping
from ray.rllib.algorithms.impala.impala_learner import (
from ray.rllib.core.rl_module.marl_module import ModuleID
from ray.rllib.utils.annotations import override
from ray.rllib.utils.lambda_defaultdict import LambdaDefaultDict
from ray.rllib.utils.metrics import LAST_TARGET_UPDATE_TS, NUM_TARGET_UPDATES
from ray.rllib.utils.schedules.scheduler import Scheduler
Dynamically update the KL loss coefficients of each module with.

        The update is completed using the mean KL divergence between the action
        distributions current policy and old policy of each module. That action
        distribution is computed during the most recent update/call to `compute_loss`.

        Args:
            module_id: The module whose KL loss coefficient to update.
            hps: The hyperparameters specific to the given `module_id`.
            sampled_kl: The computed KL loss for the given Module
                (KL divergence between the action distributions of the current
                (most recently updated) module and the old module version).
        
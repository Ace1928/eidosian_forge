import logging
from typing import Optional, Type
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig, NotProvided
from ray.rllib.algorithms.cql.cql_tf_policy import CQLTFPolicy
from ray.rllib.algorithms.cql.cql_torch_policy import CQLTorchPolicy
from ray.rllib.algorithms.sac.sac import (
from ray.rllib.execution.rollout_ops import (
from ray.rllib.execution.train_ops import (
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.annotations import override
from ray.rllib.utils.deprecation import (
from ray.rllib.utils.framework import try_import_tf, try_import_tfp
from ray.rllib.utils.metrics import (
from ray.rllib.utils.typing import ResultDict
Sets the training-related configuration.

        Args:
            bc_iters: Number of iterations with Behavior Cloning pretraining.
            temperature: CQL loss temperature.
            num_actions: Number of actions to sample for CQL loss
            lagrangian: Whether to use the Lagrangian for Alpha Prime (in CQL loss).
            lagrangian_thresh: Lagrangian threshold.
            min_q_weight: in Q weight multiplier.

        Returns:
            This updated AlgorithmConfig object.
        
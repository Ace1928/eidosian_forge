from typing import Type, Union
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.algorithms.bc.bc_catalog import BCCatalog
from ray.rllib.algorithms.marwil.marwil import MARWIL, MARWILConfig
from ray.rllib.core.learner import Learner
from ray.rllib.core.learner.learner_group_config import ModuleSpec
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.execution.rollout_ops import synchronous_parallel_sample
from ray.rllib.utils.annotations import override
from ray.rllib.utils.metrics import (
from ray.rllib.utils.typing import ResultDict
Behavioral Cloning (derived from MARWIL).

    Simply uses MARWIL with beta force-set to 0.0.
    
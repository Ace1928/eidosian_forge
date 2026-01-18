from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import numpy as np
import tree  # pip install dm_tree
from ray.rllib.core.learner.learner import (
from ray.rllib.core.rl_module.rl_module import ModuleID
from ray.rllib.utils.annotations import override
from ray.rllib.utils.lambda_defaultdict import LambdaDefaultDict
from ray.rllib.utils.metrics import (
from ray.rllib.utils.schedules.scheduler import Scheduler
from ray.rllib.utils.typing import ResultDict
def _reduce_impala_results(results: List[ResultDict]) -> ResultDict:
    """Reduce/Aggregate a list of results from Impala Learners.

    Average the values of the result dicts. Add keys for the number of agent and env
    steps trained (on all modules).

    Args:
        results: result dicts to reduce.

    Returns:
        A reduced result dict.
    """
    result = tree.map_structure(lambda *x: np.mean(x), *results)
    agent_steps_trained = sum((r[ALL_MODULES][NUM_AGENT_STEPS_TRAINED] for r in results))
    env_steps_trained = sum((r[ALL_MODULES][NUM_ENV_STEPS_TRAINED] for r in results))
    result[ALL_MODULES][NUM_AGENT_STEPS_TRAINED] = agent_steps_trained
    result[ALL_MODULES][NUM_ENV_STEPS_TRAINED] = env_steps_trained
    return result
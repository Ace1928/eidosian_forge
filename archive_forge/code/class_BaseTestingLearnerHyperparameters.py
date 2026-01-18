from typing import Any, DefaultDict, Dict, Mapping
import numpy as np
from ray.rllib.core.learner.learner import Learner, LearnerHyperparameters
from ray.rllib.core.rl_module.rl_module import ModuleID
from ray.rllib.utils.annotations import override
from ray.rllib.utils.nested_dict import NestedDict
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.typing import TensorType
class BaseTestingLearnerHyperparameters(LearnerHyperparameters):
    report_mean_weights: bool = True
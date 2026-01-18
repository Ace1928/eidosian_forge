from typing import List
import numpy as np
import tree  # pip install dm_tree
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.core.models.base import STATE_IN
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.env.single_agent_episode import SingleAgentEpisode
from ray.rllib.evaluation.postprocessing import discount_cumsum
from ray.rllib.policy.sample_batch import concat_samples, SampleBatch
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.nested_dict import NestedDict
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.torch_utils import convert_to_torch_tensor
from ray.rllib.utils.typing import TensorType
@DeveloperAPI
class Postprocessing:
    """Constant definitions for postprocessing."""
    ADVANTAGES = 'advantages'
    VALUE_TARGETS = 'value_targets'
import enum
import functools
from typing import Optional
import gymnasium as gym
import numpy as np
import tree
from gymnasium.spaces import Box, Dict, Discrete, MultiDiscrete, Tuple
from ray.rllib.core.models.base import Encoder
from ray.rllib.core.models.configs import (
from ray.rllib.core.models.configs import ModelConfig
from ray.rllib.models import MODEL_DEFAULTS
from ray.rllib.models.distributions import Distribution
from ray.rllib.models.preprocessors import get_preprocessor, Preprocessor
from ray.rllib.models.utils import get_filter_config
from ray.rllib.utils.deprecation import deprecation_warning
from ray.rllib.utils.error import UnsupportedSpaceException
from ray.rllib.utils.spaces.simplex import Simplex
from ray.rllib.utils.spaces.space_utils import flatten_space
from ray.rllib.utils.spaces.space_utils import get_base_struct_from_space
from ray.rllib.utils.typing import ViewRequirementsDict
from ray.rllib.utils.annotations import (
def _multi_action_dist_partial_helper(catalog_cls: 'Catalog', action_space: gym.Space, framework: str) -> Distribution:
    """Helper method to get a partial of a MultiActionDistribution.

    This is useful for when we want to create MultiActionDistributions from
    logits only (!) later, but know the action space now already.

    Args:
        catalog_cls: The ModelCatalog class to use.
        action_space: The action space to get the child distribution classes for.
        framework: The framework to use.

    Returns:
        A partial of the TorchMultiActionDistribution class.
    """
    action_space_struct = get_base_struct_from_space(action_space)
    flat_action_space = flatten_space(action_space)
    child_distribution_cls_struct = tree.map_structure(lambda s: catalog_cls._get_dist_cls_from_action_space(action_space=s, framework=framework), action_space_struct)
    flat_distribution_clses = tree.flatten(child_distribution_cls_struct)
    logit_lens = [int(dist_cls.required_input_dim(space)) for dist_cls, space in zip(flat_distribution_clses, flat_action_space)]
    if framework == 'torch':
        from ray.rllib.models.torch.torch_distributions import TorchMultiDistribution
        multi_action_dist_cls = TorchMultiDistribution
    elif framework == 'tf2':
        from ray.rllib.models.tf.tf_distributions import TfMultiDistribution
        multi_action_dist_cls = TfMultiDistribution
    else:
        raise ValueError(f'Unsupported framework: {framework}')
    partial_dist_cls = multi_action_dist_cls.get_partial_dist_cls(space=action_space, child_distribution_cls_struct=child_distribution_cls_struct, input_lens=logit_lens)
    return partial_dist_cls
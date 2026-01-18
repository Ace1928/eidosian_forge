from functools import partial
import gymnasium as gym
from gymnasium.spaces import Box, Dict, Discrete, MultiDiscrete, Tuple
import logging
import numpy as np
import tree  # pip install dm_tree
from typing import List, Optional, Type, Union
from ray.tune.registry import (
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.preprocessors import get_preprocessor, Preprocessor
from ray.rllib.models.tf.tf_action_dist import (
from ray.rllib.models.torch.torch_action_dist import (
from ray.rllib.utils.annotations import DeveloperAPI, PublicAPI
from ray.rllib.utils.deprecation import (
from ray.rllib.utils.error import UnsupportedSpaceException
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.from_config import from_config
from ray.rllib.utils.spaces.simplex import Simplex
from ray.rllib.utils.spaces.space_utils import flatten_space
from ray.rllib.utils.typing import ModelConfigDict, TensorType
@staticmethod
def _get_v2_model_class(input_space: gym.Space, model_config: ModelConfigDict, framework: str='tf') -> Type[ModelV2]:
    VisionNet = None
    ComplexNet = None
    if framework in ['tf2', 'tf']:
        from ray.rllib.models.tf.fcnet import FullyConnectedNetwork as FCNet
        from ray.rllib.models.tf.visionnet import VisionNetwork as VisionNet
        from ray.rllib.models.tf.complex_input_net import ComplexInputNetwork as ComplexNet
    elif framework == 'torch':
        from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as FCNet
        from ray.rllib.models.torch.visionnet import VisionNetwork as VisionNet
        from ray.rllib.models.torch.complex_input_net import ComplexInputNetwork as ComplexNet
    elif framework == 'jax':
        from ray.rllib.models.jax.fcnet import FullyConnectedNetwork as FCNet
    else:
        raise ValueError('framework={} not supported in `ModelCatalog._get_v2_model_class`!'.format(framework))
    orig_space = input_space if not hasattr(input_space, 'original_space') else input_space.original_space
    if isinstance(input_space, Box) and len(input_space.shape) == 3:
        if framework == 'jax':
            raise NotImplementedError('No non-FC default net for JAX yet!')
        return VisionNet
    elif isinstance(input_space, Box) and len(input_space.shape) == 1 and (not isinstance(orig_space, (Dict, Tuple)) or not any((isinstance(s, Box) and len(s.shape) >= 2 for s in flatten_space(orig_space)))):
        return FCNet
    else:
        if framework == 'jax':
            raise NotImplementedError('No non-FC default net for JAX yet!')
        return ComplexNet
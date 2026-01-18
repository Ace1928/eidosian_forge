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
@OverrideToImplementCustomLogic_CallToSuperRecommended
def _determine_components_hook(self):
    """Decision tree hook for subclasses to override.

        By default, this method executes the decision tree that determines the
        components that a Catalog builds. You can extend the components by overriding
        this or by adding to the constructor of your subclass.

        Override this method if you don't want to use the default components
        determined here. If you want to use them but add additional components, you
        should call `super()._determine_components()` at the beginning of your
        implementation.

        This makes it so that subclasses are not forced to create an encoder config
        if the rest of their catalog is not dependent on it or if it breaks.
        At the end of this method, an attribute `Catalog.latent_dims`
        should be set so that heads can be built using that information.
        """
    self._encoder_config = self._get_encoder_config(observation_space=self.observation_space, action_space=self.action_space, model_config_dict=self._model_config_dict, view_requirements=self._view_requirements)
    self._action_dist_class_fn = functools.partial(self._get_dist_cls_from_action_space, action_space=self.action_space)
    self.latent_dims = self._encoder_config.output_dims
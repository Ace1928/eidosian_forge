import abc
import datetime
import json
import pathlib
from dataclasses import dataclass
from typing import Mapping, Any, TYPE_CHECKING, Optional, Type, Dict, Union
import gymnasium as gym
import tree
import ray
from ray.rllib.utils.annotations import (
from ray.rllib.utils.typing import ViewRequirementsDict
from ray.rllib.utils.annotations import OverrideToImplementCustomLogic
from ray.rllib.core.models.base import STATE_IN, STATE_OUT
from ray.rllib.policy.policy import get_gym_space_from_struct_of_tensors
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.core.models.specs.typing import SpecType
from ray.rllib.core.models.specs.checker import (
from ray.rllib.models.distributions import Distribution
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID, SampleBatch
from ray.rllib.utils.nested_dict import NestedDict
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.typing import SampleBatchType
from ray.rllib.utils.serialization import (
@ExperimentalAPI
@dataclass
class RLModuleConfig:
    """A utility config class to make it constructing RLModules easier.

    Args:
        observation_space: The observation space of the RLModule. This may differ
            from the observation space of the environment. For example, a discrete
            observation space of an environment, would usually correspond to a
            one-hot encoded observation space of the RLModule because of preprocessing.
        action_space: The action space of the RLModule.
        model_config_dict: The model config dict to use.
        catalog_class: The Catalog class to use.
    """
    observation_space: gym.Space = None
    action_space: gym.Space = None
    model_config_dict: Dict[str, Any] = None
    catalog_class: Type['Catalog'] = None

    def get_catalog(self) -> 'Catalog':
        """Returns the catalog for this config."""
        return self.catalog_class(observation_space=self.observation_space, action_space=self.action_space, model_config_dict=self.model_config_dict)

    def to_dict(self):
        """Returns a serialized representation of the config.

        NOTE: This should be JSON-able. Users can test this by calling
            json.dumps(config.to_dict()).

        """
        catalog_class_path = serialize_type(self.catalog_class) if self.catalog_class else ''
        return {'observation_space': gym_space_to_dict(self.observation_space), 'action_space': gym_space_to_dict(self.action_space), 'model_config_dict': self.model_config_dict, 'catalog_class_path': catalog_class_path}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]):
        """Creates a config from a serialized representation."""
        catalog_class = None if d['catalog_class_path'] == '' else deserialize_type(d['catalog_class_path'])
        return cls(observation_space=gym_space_from_dict(d['observation_space']), action_space=gym_space_from_dict(d['action_space']), model_config_dict=d['model_config_dict'], catalog_class=catalog_class)
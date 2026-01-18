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
class SingleAgentRLModuleSpec:
    """Utility spec class to make constructing RLModules (in single-agent case) easier.

    Args:
        module_class: The RLModule class to use.
        observation_space: The observation space of the RLModule. This may differ
            from the observation space of the environment. For example, a discrete
            observation space of an environment, would usually correspond to a
            one-hot encoded observation space of the RLModule because of preprocessing.
        action_space: The action space of the RLModule.
        model_config_dict: The model config dict to use.
        catalog_class: The Catalog class to use.
        load_state_path: The path to the module state to load from. NOTE: This must be
            an absolute path.
    """
    module_class: Optional[Type['RLModule']] = None
    observation_space: Optional[gym.Space] = None
    action_space: Optional[gym.Space] = None
    model_config_dict: Optional[Dict[str, Any]] = None
    catalog_class: Optional[Type['Catalog']] = None
    load_state_path: Optional[str] = None

    def get_rl_module_config(self) -> 'RLModuleConfig':
        """Returns the RLModule config for this spec."""
        return RLModuleConfig(observation_space=self.observation_space, action_space=self.action_space, model_config_dict=self.model_config_dict, catalog_class=self.catalog_class)

    def build(self) -> 'RLModule':
        """Builds the RLModule from this spec."""
        if self.module_class is None:
            raise ValueError('RLModule class is not set.')
        if self.observation_space is None:
            raise ValueError('Observation space is not set.')
        if self.action_space is None:
            raise ValueError('Action space is not set.')
        if self.model_config_dict is None:
            raise ValueError('Model config is not set.')
        module_config = self.get_rl_module_config()
        module = self.module_class(module_config)
        return module

    @classmethod
    def from_module(cls, module: 'RLModule') -> 'SingleAgentRLModuleSpec':
        from ray.rllib.core.rl_module.marl_module import MultiAgentRLModule
        if isinstance(module, MultiAgentRLModule):
            raise ValueError('MultiAgentRLModule cannot be converted to SingleAgentRLModuleSpec.')
        return SingleAgentRLModuleSpec(module_class=type(module), observation_space=module.config.observation_space, action_space=module.config.action_space, model_config_dict=module.config.model_config_dict, catalog_class=module.config.catalog_class)

    def to_dict(self):
        """Returns a serialized representation of the spec."""
        return {'module_class': serialize_type(self.module_class), 'module_config': self.get_rl_module_config().to_dict()}

    @classmethod
    def from_dict(cls, d):
        """Returns a single agent RLModule spec from a serialized representation."""
        module_class = deserialize_type(d['module_class'])
        module_config = RLModuleConfig.from_dict(d['module_config'])
        observation_space = module_config.observation_space
        action_space = module_config.action_space
        model_config_dict = module_config.model_config_dict
        catalog_class = module_config.catalog_class
        spec = SingleAgentRLModuleSpec(module_class=module_class, observation_space=observation_space, action_space=action_space, model_config_dict=model_config_dict, catalog_class=catalog_class)
        return spec

    def update(self, other) -> None:
        """Updates this spec with the given other spec. Works like dict.update()."""
        if not isinstance(other, SingleAgentRLModuleSpec):
            raise ValueError('Can only update with another SingleAgentRLModuleSpec.')
        self.module_class = other.module_class or self.module_class
        self.observation_space = other.observation_space or self.observation_space
        self.action_space = other.action_space or self.action_space
        self.model_config_dict = other.model_config_dict or self.model_config_dict
        self.catalog_class = other.catalog_class or self.catalog_class
        self.load_state_path = other.load_state_path or self.load_state_path
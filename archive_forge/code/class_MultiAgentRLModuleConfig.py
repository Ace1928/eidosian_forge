from dataclasses import dataclass, field
import pathlib
import pprint
from typing import (
from ray.util.annotations import PublicAPI
from ray.rllib.utils.annotations import override, ExperimentalAPI
from ray.rllib.utils.nested_dict import NestedDict
from ray.rllib.core.models.specs.typing import SpecType
from ray.rllib.policy.sample_batch import MultiAgentBatch
from ray.rllib.core.rl_module.rl_module import (
from ray.rllib.utils.annotations import OverrideToImplementCustomLogic
from ray.rllib.utils.policy import validate_policy_id
from ray.rllib.utils.serialization import serialize_type, deserialize_type
from ray.rllib.utils.typing import T
@ExperimentalAPI
@dataclass
class MultiAgentRLModuleConfig:
    modules: Mapping[ModuleID, SingleAgentRLModuleSpec] = field(default_factory=dict)

    def to_dict(self):
        return {'modules': {module_id: module_spec.to_dict() for module_id, module_spec in self.modules.items()}}

    @classmethod
    def from_dict(cls, d) -> 'MultiAgentRLModuleConfig':
        return cls(modules={module_id: SingleAgentRLModuleSpec.from_dict(module_spec) for module_id, module_spec in d['modules'].items()})
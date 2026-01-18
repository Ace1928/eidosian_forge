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
@PublicAPI(stability='alpha')
@dataclass
class MultiAgentRLModuleSpec:
    """A utility spec class to make it constructing MARL modules easier.


    Users can extend this class to modify the behavior of base class. For example to
    share neural networks across the modules, the build method can be overriden to
    create the shared module first and then pass it to custom module classes that would
    then use it as a shared module.

    Args:
        marl_module_class: The class of the multi-agent RLModule to construct. By
            default it is set to MultiAgentRLModule class. This class simply loops
            throught each module and calls their foward methods.
        module_specs: The module specs for each individual module. It can be either a
            SingleAgentRLModuleSpec used for all module_ids or a dictionary mapping
            from module IDs to SingleAgentRLModuleSpecs for each individual module.
        load_state_path: The path to the module state to load from. NOTE: This must be
            an absolute path. NOTE: If the load_state_path of this spec is set, and
            the load_state_path of one of the SingleAgentRLModuleSpecs' is also set,
            the weights of that RL Module will be loaded from the path specified in
            the SingleAgentRLModuleSpec. This is useful if you want to load the weights
            of a MARL module and also manually load the weights of some of the RL
            modules within that MARL module from other checkpoints.
        modules_to_load: A set of module ids to load from the checkpoint. This is
            only used if load_state_path is set. If this is None, all modules are
            loaded.
    """
    marl_module_class: Type[MultiAgentRLModule] = MultiAgentRLModule
    module_specs: Union[SingleAgentRLModuleSpec, Dict[ModuleID, SingleAgentRLModuleSpec]] = None
    load_state_path: Optional[str] = None
    modules_to_load: Optional[Set[ModuleID]] = None

    def __post_init__(self):
        if self.module_specs is None:
            raise ValueError('Module_specs cannot be None. It should be either a SingleAgentRLModuleSpec or a dictionary mapping from module IDs to SingleAgentRLModuleSpecs for each individual module.')

    def get_marl_config(self) -> 'MultiAgentRLModuleConfig':
        """Returns the MultiAgentRLModuleConfig for this spec."""
        return MultiAgentRLModuleConfig(modules=self.module_specs)

    @OverrideToImplementCustomLogic
    def build(self, module_id: Optional[ModuleID]=None) -> Union[SingleAgentRLModuleSpec, 'MultiAgentRLModule']:
        """Builds either the multi-agent module or the single-agent module.

        If module_id is None, it builds the multi-agent module. Otherwise, it builds
        the single-agent module with the given module_id.

        Note: If when build is called the module_specs is not a dictionary, it will
        raise an error, since it should have been updated by the caller to inform us
        about the module_ids.

        Args:
            module_id: The module_id of the single-agent module to build. If None, it
                builds the multi-agent module.

        Returns:
            The built module. If module_id is None, it returns the multi-agent module.
        """
        self._check_before_build()
        if module_id:
            return self.module_specs[module_id].build()
        module_config = self.get_marl_config()
        module = self.marl_module_class(module_config)
        return module

    def add_modules(self, module_specs: Dict[ModuleID, SingleAgentRLModuleSpec], overwrite: bool=True) -> None:
        """Add new module specs to the spec or updates existing ones.

        Args:
            module_specs: The mapping for the module_id to the single-agent module
                specs to be added to this multi-agent module spec.
            overwrite: Whether to overwrite the existing module specs if they already
                exist. If False, they will be updated only.
        """
        if self.module_specs is None:
            self.module_specs = {}
        for module_id, module_spec in module_specs.items():
            if overwrite or module_id not in self.module_specs:
                self.module_specs[module_id] = module_spec
            else:
                self.module_specs[module_id].update(module_spec)

    @classmethod
    def from_module(self, module: MultiAgentRLModule) -> 'MultiAgentRLModuleSpec':
        """Creates a MultiAgentRLModuleSpec from a MultiAgentRLModule.

        Args:
            module: The MultiAgentRLModule to create the spec from.

        Returns:
            The MultiAgentRLModuleSpec.
        """
        module_specs = {module_id: SingleAgentRLModuleSpec.from_module(rl_module.unwrapped()) for module_id, rl_module in module._rl_modules.items()}
        marl_module_class = module.__class__
        return MultiAgentRLModuleSpec(marl_module_class=marl_module_class, module_specs=module_specs)

    def _check_before_build(self):
        if not isinstance(self.module_specs, dict):
            raise ValueError(f'When build() is called on {self.__class__}, the module_specs should be a dictionary mapping from module IDs to SingleAgentRLModuleSpecs for each individual module.')

    def to_dict(self) -> Dict[str, Any]:
        """Converts the MultiAgentRLModuleSpec to a dictionary."""
        return {'marl_module_class': serialize_type(self.marl_module_class), 'module_specs': {module_id: module_spec.to_dict() for module_id, module_spec in self.module_specs.items()}}

    @classmethod
    def from_dict(cls, d) -> 'MultiAgentRLModuleSpec':
        """Creates a MultiAgentRLModuleSpec from a dictionary."""
        return MultiAgentRLModuleSpec(marl_module_class=deserialize_type(d['marl_module_class']), module_specs={module_id: SingleAgentRLModuleSpec.from_dict(module_spec) for module_id, module_spec in d['module_specs'].items()})

    def update(self, other: 'MultiAgentRLModuleSpec', overwrite=False) -> None:
        """Updates this spec with the other spec.

        Traverses this MultiAgentRLModuleSpec's module_specs and updates them with
        the module specs from the other MultiAgentRLModuleSpec.

        Args:
            other: The other spec to update this spec with.
            overwrite: Whether to overwrite the existing module specs if they already
                exist. If False, they will be updated only.
        """
        assert type(other) is MultiAgentRLModuleSpec
        if isinstance(other.module_specs, dict):
            self.add_modules(other.module_specs, overwrite=overwrite)
        elif not self.module_specs:
            self.module_specs = other.module_specs
        else:
            self.module_specs.update(other.module_specs)
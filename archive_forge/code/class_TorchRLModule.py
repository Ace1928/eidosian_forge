import pathlib
from typing import Any, List, Mapping, Tuple, Union, Type
from packaging import version
from ray.rllib.core.rl_module import RLModule
from ray.rllib.core.rl_module.rl_module_with_target_networks_interface import (
from ray.rllib.core.rl_module.torch.torch_compile_config import TorchCompileConfig
from ray.rllib.models.torch.torch_distributions import TorchDistribution
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_utils import TORCH_COMPILE_REQUIRED_VERSION
from ray.rllib.utils.typing import NetworkType
class TorchRLModule(nn.Module, RLModule):
    """A base class for RLlib PyTorch RLModules.

    Note that the `_forward` methods of this class can be 'torch.compiled' individually:
        - `TorchRLModule._forward_train()`
        - `TorchRLModule._forward_inference()`
        - `TorchRLModule._forward_exploration()`

    As a rule of thumb, they should only contain torch-native tensor manipulations,
    or otherwise they may yield wrong outputs. In particular, the creation of RLlib
    distributions inside these methods should be avoided when using `torch.compile`.
    When in doubt, you can use `torch.dynamo.explain()` to check whether a compiled
    method has broken up into multiple sub-graphs.

    Compiling these methods can bring speedups under certain conditions.
    """
    framework: str = 'torch'

    def __init__(self, *args, **kwargs) -> None:
        nn.Module.__init__(self)
        RLModule.__init__(self, *args, **kwargs)

    def forward(self, batch: Mapping[str, Any], **kwargs) -> Mapping[str, Any]:
        """forward pass of the module.

        This is aliased to forward_train because Torch DDP requires a forward method to
        be implemented for backpropagation to work.
        """
        return self.forward_train(batch, **kwargs)

    def compile(self, compile_config: TorchCompileConfig):
        """Compile the forward methods of this module.

        This is a convenience method that calls `compile_wrapper` with the given
        compile_config.

        Args:
            compile_config: The compile config to use.
        """
        return compile_wrapper(self, compile_config)

    @override(RLModule)
    def get_state(self) -> Mapping[str, Any]:
        return self.state_dict()

    @override(RLModule)
    def set_state(self, state_dict: Mapping[str, Any]) -> None:
        self.load_state_dict(state_dict)

    def _module_state_file_name(self) -> pathlib.Path:
        return pathlib.Path('module_state.pt')

    @override(RLModule)
    def save_state(self, dir: Union[str, pathlib.Path]) -> None:
        path = str(pathlib.Path(dir) / self._module_state_file_name())
        torch.save(self.state_dict(), path)

    @override(RLModule)
    def load_state(self, dir: Union[str, pathlib.Path]) -> None:
        path = str(pathlib.Path(dir) / self._module_state_file_name())
        self.set_state(torch.load(path))
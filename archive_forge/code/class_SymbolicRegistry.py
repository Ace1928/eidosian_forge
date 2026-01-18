import warnings
from typing import (
from torch.onnx import _constants, errors
from torch.onnx._internal import _beartype
class SymbolicRegistry:
    """Registry for symbolic functions.

    The registry maintains a mapping from qualified names to symbolic functions.
    It is used to register new symbolic functions and to dispatch calls to
    the appropriate function.
    """

    def __init__(self) -> None:
        self._registry: Dict[str, _SymbolicFunctionGroup] = {}

    def register(self, name: str, opset: OpsetVersion, func: Callable, custom: bool=False) -> None:
        """Registers a symbolic function.

        Args:
            name: The qualified name of the function to register. In the form of 'domain::op'.
                E.g. 'aten::add'.
            opset: The opset version of the function to register.
            func: The symbolic function to register.
            custom: Whether the function is a custom function that overrides existing ones.

        Raises:
            ValueError: If the separator '::' is not in the name.
        """
        if '::' not in name:
            raise ValueError(f"The name must be in the form of 'domain::op', not '{name}'")
        symbolic_functions = self._registry.setdefault(name, _SymbolicFunctionGroup(name))
        if custom:
            symbolic_functions.add_custom(func, opset)
        else:
            symbolic_functions.add(func, opset)

    def unregister(self, name: str, opset: OpsetVersion) -> None:
        """Unregisters a symbolic function.

        Args:
            name: The qualified name of the function to unregister.
            opset: The opset version of the function to unregister.
        """
        if name not in self._registry:
            return
        self._registry[name].remove_custom(opset)

    def get_function_group(self, name: str) -> Optional[_SymbolicFunctionGroup]:
        """Returns the function group for the given name."""
        return self._registry.get(name)

    def is_registered_op(self, name: str, version: int) -> bool:
        """Returns whether the given op is registered for the given opset version."""
        functions = self.get_function_group(name)
        if functions is None:
            return False
        return functions.get(version) is not None

    def all_functions(self) -> Set[str]:
        """Returns the set of all registered function names."""
        return set(self._registry)
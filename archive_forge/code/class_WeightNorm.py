from torch.nn.parameter import Parameter, UninitializedParameter
from torch import _weight_norm, norm_except_dim
from typing import Any, TypeVar
import warnings
from ..modules import Module
class WeightNorm:
    name: str
    dim: int

    def __init__(self, name: str, dim: int) -> None:
        if dim is None:
            dim = -1
        self.name = name
        self.dim = dim

    def compute_weight(self, module: Module) -> Any:
        g = getattr(module, self.name + '_g')
        v = getattr(module, self.name + '_v')
        return _weight_norm(v, g, self.dim)

    @staticmethod
    def apply(module, name: str, dim: int) -> 'WeightNorm':
        warnings.warn('torch.nn.utils.weight_norm is deprecated in favor of torch.nn.utils.parametrizations.weight_norm.')
        for hook in module._forward_pre_hooks.values():
            if isinstance(hook, WeightNorm) and hook.name == name:
                raise RuntimeError(f'Cannot register two weight_norm hooks on the same parameter {name}')
        if dim is None:
            dim = -1
        fn = WeightNorm(name, dim)
        weight = getattr(module, name)
        if isinstance(weight, UninitializedParameter):
            raise ValueError("The module passed to `WeightNorm` can't have uninitialized parameters. Make sure to run the dummy forward before applying weight normalization")
        del module._parameters[name]
        module.register_parameter(name + '_g', Parameter(norm_except_dim(weight, 2, dim).data))
        module.register_parameter(name + '_v', Parameter(weight.data))
        setattr(module, name, fn.compute_weight(module))
        module.register_forward_pre_hook(fn)
        return fn

    def remove(self, module: Module) -> None:
        weight = self.compute_weight(module)
        delattr(module, self.name)
        del module._parameters[self.name + '_g']
        del module._parameters[self.name + '_v']
        setattr(module, self.name, Parameter(weight.data))

    def __call__(self, module: Module, inputs: Any) -> None:
        setattr(module, self.name, self.compute_weight(module))
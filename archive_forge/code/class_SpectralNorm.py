import torch
from torch.nn.functional import normalize
from typing import Any, Optional, TypeVar
from ..modules import Module
class SpectralNorm:
    _version: int = 1
    name: str
    dim: int
    n_power_iterations: int
    eps: float

    def __init__(self, name: str='weight', n_power_iterations: int=1, dim: int=0, eps: float=1e-12) -> None:
        self.name = name
        self.dim = dim
        if n_power_iterations <= 0:
            raise ValueError(f'Expected n_power_iterations to be positive, but got n_power_iterations={n_power_iterations}')
        self.n_power_iterations = n_power_iterations
        self.eps = eps

    def reshape_weight_to_matrix(self, weight: torch.Tensor) -> torch.Tensor:
        weight_mat = weight
        if self.dim != 0:
            weight_mat = weight_mat.permute(self.dim, *[d for d in range(weight_mat.dim()) if d != self.dim])
        height = weight_mat.size(0)
        return weight_mat.reshape(height, -1)

    def compute_weight(self, module: Module, do_power_iteration: bool) -> torch.Tensor:
        weight = getattr(module, self.name + '_orig')
        u = getattr(module, self.name + '_u')
        v = getattr(module, self.name + '_v')
        weight_mat = self.reshape_weight_to_matrix(weight)
        if do_power_iteration:
            with torch.no_grad():
                for _ in range(self.n_power_iterations):
                    v = normalize(torch.mv(weight_mat.t(), u), dim=0, eps=self.eps, out=v)
                    u = normalize(torch.mv(weight_mat, v), dim=0, eps=self.eps, out=u)
                if self.n_power_iterations > 0:
                    u = u.clone(memory_format=torch.contiguous_format)
                    v = v.clone(memory_format=torch.contiguous_format)
        sigma = torch.dot(u, torch.mv(weight_mat, v))
        weight = weight / sigma
        return weight

    def remove(self, module: Module) -> None:
        with torch.no_grad():
            weight = self.compute_weight(module, do_power_iteration=False)
        delattr(module, self.name)
        delattr(module, self.name + '_u')
        delattr(module, self.name + '_v')
        delattr(module, self.name + '_orig')
        module.register_parameter(self.name, torch.nn.Parameter(weight.detach()))

    def __call__(self, module: Module, inputs: Any) -> None:
        setattr(module, self.name, self.compute_weight(module, do_power_iteration=module.training))

    def _solve_v_and_rescale(self, weight_mat, u, target_sigma):
        v = torch.linalg.multi_dot([weight_mat.t().mm(weight_mat).pinverse(), weight_mat.t(), u.unsqueeze(1)]).squeeze(1)
        return v.mul_(target_sigma / torch.dot(u, torch.mv(weight_mat, v)))

    @staticmethod
    def apply(module: Module, name: str, n_power_iterations: int, dim: int, eps: float) -> 'SpectralNorm':
        for hook in module._forward_pre_hooks.values():
            if isinstance(hook, SpectralNorm) and hook.name == name:
                raise RuntimeError(f'Cannot register two spectral_norm hooks on the same parameter {name}')
        fn = SpectralNorm(name, n_power_iterations, dim, eps)
        weight = module._parameters[name]
        if weight is None:
            raise ValueError(f'`SpectralNorm` cannot be applied as parameter `{name}` is None')
        if isinstance(weight, torch.nn.parameter.UninitializedParameter):
            raise ValueError("The module passed to `SpectralNorm` can't have uninitialized parameters. Make sure to run the dummy forward before applying spectral normalization")
        with torch.no_grad():
            weight_mat = fn.reshape_weight_to_matrix(weight)
            h, w = weight_mat.size()
            u = normalize(weight.new_empty(h).normal_(0, 1), dim=0, eps=fn.eps)
            v = normalize(weight.new_empty(w).normal_(0, 1), dim=0, eps=fn.eps)
        delattr(module, fn.name)
        module.register_parameter(fn.name + '_orig', weight)
        setattr(module, fn.name, weight.data)
        module.register_buffer(fn.name + '_u', u)
        module.register_buffer(fn.name + '_v', v)
        module.register_forward_pre_hook(fn)
        module._register_state_dict_hook(SpectralNormStateDictHook(fn))
        module._register_load_state_dict_pre_hook(SpectralNormLoadStateDictPreHook(fn))
        return fn
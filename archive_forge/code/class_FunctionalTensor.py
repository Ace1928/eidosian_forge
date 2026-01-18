import contextlib
from abc import ABC, abstractmethod
from typing import Any, Callable, ContextManager, Tuple
import torch
import torch.utils._pytree as pytree
from torch._C import _functionalization_reapply_views_tls as _reapply_views
from torch.utils._python_dispatch import return_and_correct_aliasing, TorchDispatchMode
class FunctionalTensor(torch.Tensor):
    """
    Functional tensors represent tensors that will remove mutations
    from a program. If you perform a mutable operation on a functional tensor,
    it will re-dispatch to the functional variant of that operation.

    Historically, functionalization is implemented in C++ in the dispatcher.
    This class is a lightweight python shim around the C++ functionalization logic.

    FunctionalTensor is required to be used with a corresponding
    FunctionalTensormode active, because it relies
    on using the mode for dispatch (which can properly handle factory functions).
    """
    elem: torch.Tensor
    _mode_key = torch._C._TorchDispatchModeKey.FUNCTIONAL
    _extra_dispatch_keys = torch._C._additional_keys_to_prop_for_wrapper_tensors.add(torch._C.DispatchKey.ZeroTensor)
    metadata_fns = [torch.ops.aten.is_contiguous.default, torch.ops.aten.is_contiguous.memory_format, torch.ops.aten.is_strides_like_format.default, torch.ops.aten.is_non_overlapping_and_dense.default, torch.ops.aten.size.default, torch.ops.aten.sym_size.default, torch.ops.aten.stride.default, torch.ops.aten.sym_stride.default, torch.ops.aten.storage_offset.default, torch.ops.aten.sym_storage_offset.default, torch.ops.aten.numel.default, torch.ops.aten.sym_numel.default, torch.ops.aten.dim.default]

    def __new__(cls, elem):
        assert torch._is_functional_tensor(elem)
        extra_dispatch_keys = FunctionalTensor._extra_dispatch_keys & torch._C._dispatch_keys(elem)
        out = torch.Tensor._make_wrapper_subclass(cls, elem.shape, elem.stride(), elem.storage_offset(), None, elem.dtype, elem.layout, elem.device, False, elem.requires_grad, 'sizes', False, False, extra_dispatch_keys)
        out.elem = elem
        return out
    __torch_function__ = torch._C._disabled_torch_function_impl

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        unrecognized_types = [t for t in types if t not in [torch.Tensor, torch._subclasses.FakeTensor, FunctionalTensor]]
        if unrecognized_types:
            not_implemented_log.debug('FunctionalTensor unrecognized subclass(es): %s', unrecognized_types)
            return NotImplemented
        if kwargs is None:
            kwargs = {}
        if func in FunctionalTensor.metadata_fns:

            def unwrap(x):
                return x.elem
            assert len(args) == 1 and isinstance(args[0], FunctionalTensor)
            assert len(kwargs) == 0
            return func(args[0].elem)
        raise RuntimeError('Attempting to use FunctionalTensor on its own. Instead, please use it with a corresponding FunctionalTensorMode()')

    def __repr__(self):
        return f'FunctionalTensor({repr(self.elem)})'

    @staticmethod
    def to_functional(x):
        assert not torch._is_functional_tensor(x)
        x_functional = torch._to_functional_tensor(x)
        with FunctionalTensorMode():
            torch._mirror_autograd_meta_to(x, x_functional)
            out = FunctionalTensor(x_functional)
            torch._mirror_autograd_meta_to(x_functional, out)
        return out

    def from_functional(self):
        torch._sync(self)
        return torch._from_functional_tensor(self.elem)

    def replace_(self, output) -> None:
        torch._functionalize_replace(self.elem, output)

    def commit_update(self) -> None:
        torch._functionalize_commit_update(self.elem)

    def sync(self) -> None:
        torch._functionalize_sync(self.elem)

    def mark_mutation_hidden_from_autograd(self) -> None:
        torch._functionalize_mark_mutation_hidden_from_autograd(self.elem)
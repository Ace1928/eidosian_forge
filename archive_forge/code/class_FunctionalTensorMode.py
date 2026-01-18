import contextlib
from abc import ABC, abstractmethod
from typing import Any, Callable, ContextManager, Tuple
import torch
import torch.utils._pytree as pytree
from torch._C import _functionalization_reapply_views_tls as _reapply_views
from torch.utils._python_dispatch import return_and_correct_aliasing, TorchDispatchMode
class FunctionalTensorMode(TorchDispatchMode):

    def __init__(self):
        self.is_on_stack = False
        self.enter_stack = []
        self._mode_key = torch._C._TorchDispatchModeKey.FUNCTIONAL
        self.decompose_composite_implicit_ops = True

    def __enter__(self):
        if torch._C._get_dispatch_mode(torch._C._TorchDispatchModeKey.FUNCTIONAL) is None:
            self.enter_stack.append(True)
            return super().__enter__()
        else:
            self.enter_stack.append(False)
            return self

    def __exit__(self, a, b, c):
        is_on_stack = self.enter_stack.pop()
        if is_on_stack:
            super().__exit__(a, b, c)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        unrecognized_types = [t for t in types if not issubclass(t, torch._subclasses.FakeTensor) and t not in [torch.Tensor, FunctionalTensor]]
        if unrecognized_types:
            not_implemented_log.debug('FunctionalTensor unrecognized subclass(es): %s', unrecognized_types)
            return NotImplemented
        if func not in FunctionalTensor.metadata_fns and self.decompose_composite_implicit_ops and torch._C._dispatch_has_kernel(func.name()):
            with self:
                r = func.decompose(*args, **kwargs)
                if r is not NotImplemented:
                    return r

        def assert_is_functional(x):
            assert torch._is_functional_tensor(x)

        def wrap(x):
            assert not isinstance(x, FunctionalTensor)
            if isinstance(x, torch.Tensor) and torch._is_functional_tensor(x):
                return FunctionalTensor(x)
            return x
        any_functional_inputs = False

        def unwrap(x):
            any_functional_inputs = True
            return x.elem
        from torch._higher_order_ops.auto_functionalize import can_auto_functionalize, do_auto_functionalize
        if can_auto_functionalize(func) and (not torch._C._dispatch_has_kernel_for_dispatch_key(func.name(), torch._C.DispatchKey.Functionalize)):
            return do_auto_functionalize(func, args, kwargs)
        args_unwrapped, kwargs_unwrapped = pytree.tree_map_only(FunctionalTensor, unwrap, (args, kwargs))
        is_included = torch._C._dispatch_tls_is_dispatch_key_included(torch._C.DispatchKey.Functionalize)
        is_excluded = torch._C._dispatch_tls_is_dispatch_key_excluded(torch._C.DispatchKey.Functionalize)
        assert is_excluded or not is_included
        include_to_set = torch._C._dispatch_tls_local_include_set() | torch._C.DispatchKeySet(torch._C.DispatchKey.Functionalize)
        exclude_to_set = torch._C._dispatch_tls_local_exclude_set().remove(torch._C.DispatchKey.Functionalize) - FunctionalTensor._extra_dispatch_keys
        with torch._C._ForceDispatchKeyGuard(include_to_set, exclude_to_set):
            try:
                old_apply_views = torch._functionalize_enable_reapply_views(True)
                outs_unwrapped = func(*args_unwrapped, **kwargs_unwrapped)
                outs_wrapped = pytree.tree_map_only(torch.Tensor, wrap, outs_unwrapped)
            finally:
                torch._disable_functionalization()
                torch._functionalize_enable_reapply_views(old_apply_views)
        is_included = torch._C._dispatch_tls_is_dispatch_key_included(torch._C.DispatchKey.Functionalize)
        is_excluded = torch._C._dispatch_tls_is_dispatch_key_excluded(torch._C.DispatchKey.Functionalize)
        assert is_excluded or not is_included
        if not any((isinstance(x, FunctionalTensor) for x in pytree.tree_leaves(outs_wrapped))) or func == torch.ops.aten.lift_fresh.default:
            return outs_wrapped
        return return_and_correct_aliasing(func, args, kwargs, outs_wrapped)
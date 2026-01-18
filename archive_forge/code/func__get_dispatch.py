import contextlib
import ctypes
import importlib
import inspect
import sys
import types
from typing import Any, Callable, Dict, List, Type, Union
import torch._C
import torch.utils._pytree as pytree
from torch import _utils_internal
from torch._functorch.pyfunctorch import dispatch_functorch
def _get_dispatch(self, key):
    assert key not in self._dispatch_cache, f'{self} {key}'
    if key == torch._C.DispatchKey.Python:
        if not self.python_key_mode_table:
            self._dispatch_cache[key] = key
            add_cached_op(self)
            return key

        def handler(*args, **kwargs):
            from torch.utils._python_dispatch import _get_current_dispatch_mode
            curr_mode = type(_get_current_dispatch_mode())
            assert curr_mode is not None, 'Illegal invocation of dispatch on torch._C.DispatchKey.Python without a mode.'
            if curr_mode not in self.python_key_mode_table:
                return self._op_dk(key, *args, **kwargs)
            return self.python_key_mode_table[curr_mode](*args, **kwargs)
        self._dispatch_cache[key] = handler
        add_cached_op(self)
        return handler
    cache_result = True
    functionality_key = torch._C._to_functionality_key(key)
    if functionality_key in mode_stack_per_key():
        curr_stack = mode_stack_per_key()[functionality_key]
        if len(curr_stack) > 0 and (not torch._C._dispatch_tls_is_dispatch_key_excluded(DispatchKey.Python)):

            def handler(*args, **kwargs):
                with temporarily_pop_mode(curr_stack) as curr_mode:
                    assert hasattr(curr_mode, '__torch_dispatch__')
                    overload_types = []
                    args_flattened = pytree.arg_tree_leaves(*args, **kwargs)
                    for a in args_flattened:
                        if isinstance(a, torch.Tensor) and torch._C._dispatch_keys(a).has(torch._C.DispatchKey.Python):
                            overload_types.append(type(a))
                    return curr_mode.__torch_dispatch__(self, overload_types, args, kwargs)
            return handler
        else:
            cache_result = False
    final_key = resolve_key(self, key)
    if key == torch._C.DispatchKey.Functionalize:
        import torch._dispatch.python as pydispatch
        if pydispatch.CROSSREF_FUNCTIONALIZE:
            handler = pydispatch.make_crossref_functionalize(self, final_key)
            if cache_result:
                self._dispatch_cache[key] = handler
                add_cached_op(self)
            return handler
    r = self.py_kernels.get(final_key, final_key)
    if cache_result:
        self._dispatch_cache[key] = r
        add_cached_op(self)
    return r
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
class OpOverload(OperatorBase):

    def __init__(self, overloadpacket, op, op_dk, schema, tags):
        super().__init__()
        self._op = op
        self._op_dk = op_dk
        self._schema = schema
        self._overloadpacket = overloadpacket
        self._tags = tags
        self._overloadname = 'default' if schema.overload_name == '' else schema.overload_name
        self._name = self._schema.name
        if schema.overload_name:
            self._name += '.' + schema.overload_name
        self.__name__ = f'{self._schema.name.split('::')[1]}.{self._overloadname}'
        self.__module__ = overloadpacket.__module__
        op.__module__ = overloadpacket.__module__
        self.__qualname__ = self._name
        self.__annotations__ = {}
        self._defined_in_python = self.__qualname__ in torch.library._defs
        is_write = None
        for a in self._schema.arguments:
            if a.alias_info is None:
                continue
            if is_write is None:
                is_write = a.alias_info.is_write
            else:
                is_write = a.alias_info.is_write or is_write
        self.is_view = is_write is not None and (not is_write)

    def __deepcopy__(self, memo=None):
        return self

    def __repr__(self):
        return "<OpOverload(op='{}.{}', overload='{}')>".format(*self._schema.name.split('::'), self._overloadname)

    def __call__(self, *args, **kwargs):
        return self._op(*args, **kwargs or {})

    def __hash__(self):
        return hash(self._op)

    def __str__(self):
        return '{}.{}.{}'.format(*self._schema.name.split('::'), self._overloadname)

    def has_kernel_for_dispatch_key(self, k):
        return super().has_kernel_for_dispatch_key(k) or torch._C._dispatch_has_kernel_for_dispatch_key(self.name(), k)

    def has_kernel_for_any_dispatch_key(self, ks):
        return torch._C._dispatch_has_kernel_for_any_dispatch_key(self.name(), ks) or super().has_kernel_for_any_dispatch_key(ks)

    @property
    def namespace(self):
        return self._schema.name.split('::')[0]

    def decompose(self, *args, **kwargs):
        dk = torch._C.DispatchKey.CompositeImplicitAutograd
        if dk in self.py_kernels:
            return self.py_kernels[dk](*args, **kwargs)
        elif torch._C._dispatch_has_kernel_for_dispatch_key(self.name(), dk):
            return self._op_dk(dk, *args, **kwargs)
        else:
            return NotImplemented

    def _uncache_dispatch(self, key):
        self._dispatch_cache.pop(key, None)

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

    def name(self):
        return self._name

    @property
    def overloadpacket(self):
        return self._overloadpacket

    @property
    def op(self):
        return self._op

    @property
    def tags(self):
        return self._tags
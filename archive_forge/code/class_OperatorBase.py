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
class OperatorBase:
    """
    Base class for OpOverload (which represents C++ ATen operators) and HigherOrderOperator
    (which represents Python-only operators that are unrepresentable in TorchScript).
    """

    def __init__(self):
        self._dispatch_cache: Dict[torch._C.DispatchKey, Union[torch._C.DispatchKey, Callable[..., Any]]] = {}
        self.py_kernels: Dict[torch._C.DispatchKey, Callable[..., Any]] = {}
        from torch.utils._python_dispatch import TorchDispatchMode
        self.python_key_mode_table: Dict[Type[TorchDispatchMode], Callable[..., Any]] = {}
        self.functorch_table = {}

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()

    def has_kernel_for_dispatch_key(self, k):
        return k in self.py_kernels

    def has_kernel_for_any_dispatch_key(self, ks):
        for k in self.py_kernels:
            if not torch._C._dispatch_is_alias_key(k) and ks.has(k):
                return True
        return False

    def py_impl(self, k):

        def inner(fn):
            if inspect.isclass(k) and issubclass(k, torch.utils._python_dispatch.TorchDispatchMode):
                assert k not in self.python_key_mode_table
                self.python_key_mode_table[k] = fn
                self._dispatch_cache.clear()
                return fn
            if isinstance(k, torch._C._functorch.TransformType):
                assert k not in self.functorch_table
                self.functorch_table[k] = fn
                return fn
            assert isinstance(k, torch._C.DispatchKey)
            assert k != torch._C.DispatchKey.Python, 'Please register a mode for the torch._C.DispatchKey.Python key instead.'
            if k in self.py_kernels:
                raise RuntimeError(f'Trying to override a python impl for {k} on operator {self.name()}')
            self.py_kernels[k] = fn
            self._dispatch_cache.clear()
            return fn
        return inner

    def py_functionalize_impl(self, fn):
        from torch._subclasses.functional_tensor import CppFunctionalizeAPI as _CppFunctionalizeAPI, FunctorchFunctionalizeAPI as _FunctorchFunctionalizeAPI, PythonFunctionalizeAPI as _PythonFunctionalizeAPI

        def functionalize_dk_fn(*args, **kwargs):
            return fn(_CppFunctionalizeAPI(), *args, **kwargs)

        def functionalize_dispatch_mode_fn(mode, *args, **kwargs):
            return fn(_PythonFunctionalizeAPI(), *args, **kwargs)

        def functionalize_functorch_fn(interpreter, *args, **kwargs):
            return fn(_FunctorchFunctionalizeAPI(interpreter), *args, **kwargs)
        self.py_impl(torch._C.DispatchKey.Functionalize)(functionalize_dk_fn)
        self.py_impl(torch._subclasses.functional_tensor.FunctionalTensorMode)(functionalize_dispatch_mode_fn)
        self.py_impl(torch._C._functorch.TransformType.Functionalize)(functionalize_functorch_fn)
        return fn

    def name(self):
        raise NotImplementedError()
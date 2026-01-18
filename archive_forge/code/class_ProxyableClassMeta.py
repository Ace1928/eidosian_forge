import builtins
import copy
import functools
import inspect
import math
import os
import warnings
import collections
from itertools import chain
from types import CodeType, FunctionType, ModuleType
from typing import (
import torch
import torch.utils._pytree as pytree
from torch._C import ScriptObject  # type: ignore[attr-defined]
from ._compatibility import compatibility
from .graph import _PyTreeCodeGen, _PyTreeInfo, Graph
from .graph_module import GraphModule
from .node import Argument, base_types, map_aggregate
from .proxy import ParameterProxy, Proxy, TracerBase, Scope, ScopeContextManager
@compatibility(is_backward_compatible=True)
class ProxyableClassMeta(type):
    """
    ProxyableClassMeta allows you to make construction of a given Python class
    symbolically traceable. For example::

        import torch
        import torch.fx

        class TensorPair(metaclass=torch.fx.ProxyableClassMeta):
            def __init__(self, left, right):
                self.left, self.right = left, right

            def add(self, other):
                l = self.left + other.left
                r = self.right + other.right
                return TensorPair(l, r)

            def mul(self, other):
                l = self.left * other.left
                r = self.right * other.right
                return TensorPair(l, r)

        def use_tensor_pair_ctor(x : TensorPair, y : torch.Tensor):
            s = x.add(TensorPair(y, y))
            return s.mul(x)

        x = TensorPair(torch.randn(5, 3), torch.randn(5, 3))
        y = torch.randn(5, 3)
        ref_out = use_tensor_pair_ctor(x, y)

        traced = torch.fx.symbolic_trace(use_tensor_pair_ctor)
        print(traced.code)
        '''
        def forward(self, x : __main___TensorPair, y : torch.Tensor):
            tensor_pair = __main___TensorPair(y, y);  y = None
            add = x.add(tensor_pair);  tensor_pair = None
            mul = add.mul(x);  add = x = None
            return mul
        '''

    From this example, we can see that construction of a class (``TensorPair``)
    defined with ``ProxyableClassMeta`` as metaclass can be recorded in symbolic
    tracing.
    """

    def __init__(cls, name, bases, attrs):
        _proxyable_classes.setdefault(cls)
        super().__init__(name, bases, attrs)

    def __call__(cls, *args, **kwargs):
        instance = cls.__new__(cls)
        if not is_fx_tracing():
            cls.__init__(instance, *args, **kwargs)
            return instance
        found_proxies = []

        def check_proxy(a):
            if isinstance(a, Proxy):
                found_proxies.append(a)
        map_aggregate(args, check_proxy)
        map_aggregate(kwargs, check_proxy)
        if len(found_proxies) != 0:
            tracer = found_proxies[0].tracer
            return tracer.create_proxy('call_function', cls, args, kwargs)
        else:
            cls.__init__(instance, *args, **kwargs)
            return instance
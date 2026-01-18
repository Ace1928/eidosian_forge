import contextlib
import functools
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.utils._pytree as pytree
from torch.fx import Tracer, GraphModule
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode, unset_fake_temporarily, is_fake
from torch._dispatch.python import enable_python_dispatcher, enable_pre_dispatch
import torch.fx as fx
from torch.fx.passes.shape_prop import _extract_tensor_metadata
from contextlib import contextmanager, nullcontext
import inspect
from dataclasses import dataclass
import weakref
import operator
from torch.utils._stats import count
import logging
from torch.overrides import TorchFunctionMode
from torch.utils._python_dispatch import (
from .sym_node import SymNode
from ._sym_dispatch_mode import SymDispatchMode
from torch.fx import Proxy
import torch.fx.traceback as fx_traceback
from torch import SymInt, SymFloat, SymBool
from torch.utils.weak import WeakTensorKeyDictionary
class DecompositionInterpreter(torch.fx.Interpreter):

    def __init__(self, module: torch.fx.GraphModule, new_graph: torch.fx.Graph, decomposition_table=None, **kwargs):
        super().__init__(module, **kwargs)
        self.new_graph = new_graph
        self.tracer = torch.fx.proxy.GraphAppendingTracer(self.new_graph)
        self.tracer.tensor_tracker = WeakTensorKeyDictionary()
        self.tracer.symnode_tracker = weakref.WeakKeyDictionary()
        self.decomposition_table = decomposition_table
        if self.decomposition_table is None:
            self.decomposition_table = {}
        self.mode = ProxyTorchDispatchMode(self.tracer, tracing_mode='real')

    def placeholder(self, target, args, kwargs):
        out = super().placeholder(target, args, kwargs)
        proxy = torch.fx.Proxy(self.new_graph.placeholder(target), self.tracer)
        track_tensor_tree(out, proxy, constant=None, tracer=self.tracer)
        return out

    def get_attr(self, target, args, kwargs):
        out = super().get_attr(target, args, kwargs)
        proxy = torch.fx.Proxy(self.new_graph.get_attr(target), self.tracer)
        track_tensor_tree(out, proxy, constant=None, tracer=self.tracer)
        return out

    def output(self, target, args, kwargs):
        out = super().output(target, args, kwargs)

        def unwrap(e):
            return get_proxy_slot(e, self.tracer, e, lambda x: x.proxy.node)
        self.new_graph.output(pytree.tree_map(unwrap, out))
        return out

    def run(self, *args, **kwargs):
        with decompose(self.decomposition_table), self.mode:
            return super().run(*args, **kwargs)
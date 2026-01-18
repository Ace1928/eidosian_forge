import operator
import traceback
import typing
from contextlib import nullcontext
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
from functorch.experimental.control_flow import _unstack_pytree
from torch import fx
from torch._dispatch.python import enable_python_dispatcher
from torch._export.pass_infra.node_metadata import NodeMetadata
from torch._export.pass_infra.proxy_value import ProxyValue
from torch._subclasses import FakeTensor, UnsupportedFakeTensorException
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx import traceback as fx_traceback
from torch.fx.experimental.proxy_tensor import PythonKeyTracer
from torch.fx.graph import CodeGen
from torch.fx.passes.infra.pass_base import PassBase, PassResult
from torch.fx.passes.shape_prop import _extract_tensor_metadata, TensorMetadata
from torch.utils import _pytree as pytree
class ExportTracer(PythonKeyTracer):
    """
        Tracer used to create nodes during the retracing part of the Expo_ExportPassBasertPassBase
        """

    def __init__(self, callback: '_ExportPassBase', codegen: CodeGen) -> None:
        super().__init__()
        self.callback = callback
        self.root = torch.nn.Module()
        self.graph = torch.fx.Graph()
        self.graph.set_codegen(codegen)
        self.tensor_attrs: Dict[str, torch.Tensor] = {}
        self.fake_tensor_mode: Optional[FakeTensorMode] = None
        self.submodules: Dict[torch.nn.Module, str] = {}

    def trace(self) -> None:
        raise ExportPassBaseError("ExportTracer doesn't support trace().")

    def create_arg(self, a: Argument) -> torch.fx.Node:
        if isinstance(a, torch.nn.Module):
            if a not in self.submodules:
                name_submodule = f'submodule_{len(self.submodules)}'
                self.root.add_module(name_submodule, a)
                self.submodules[a] = name_submodule
        elif isinstance(a, FakeTensor):
            if not hasattr(a, 'constant') or a.constant is None:
                raise ExportPassBaseError(f'Cannot add {a} to graph.')
            a = a.constant
        node = super().create_arg(a)
        if isinstance(a, torch.Tensor) and isinstance(node, torch.fx.Node) and (node.op == 'get_attr'):
            self.set_metadata(node, a)
            self.callback.on_attr(ProxyValue(a, node))
        return node

    def set_metadata(self, node: torch.fx.Node, value: Argument) -> None:

        def make_val(x: Argument) -> Union[FakeTensor, torch.SymInt, torch.SymFloat, torch.SymBool, int, float, bool, str, None]:
            if isinstance(x, FakeTensor):
                return x
            elif isinstance(x, torch.Tensor):
                if x.is_quantized:
                    x = torch.dequantize(x)
                try:
                    assert self.fake_tensor_mode is not None
                    if isinstance(x, torch.nn.Parameter):
                        fake_tensor = self.fake_tensor_mode.from_tensor(x, static_shapes=True)
                    else:
                        fake_tensor = self.fake_tensor_mode.from_tensor(x)
                except UnsupportedFakeTensorException:
                    print('Fakeifying a Tensor subclass is not supported                             right now. Instead a TensorMetadata is used.')
                    fake_tensor = None
                return fake_tensor
            elif isinstance(x, (torch.SymInt, torch.SymFloat, torch.SymBool, int, float, bool, str)):
                return x
            else:
                return None
        node.meta['val'] = pytree.tree_map(make_val, value)

        def make_tensor_meta(x: Argument) -> Optional[TensorMetadata]:
            if not isinstance(x, FakeTensor) and isinstance(x, torch.Tensor):
                if x.is_quantized:
                    x = torch.dequantize(x)
                try:
                    assert self.fake_tensor_mode is not None
                    _ = self.fake_tensor_mode.from_tensor(x)
                    tensor_meta = None
                except UnsupportedFakeTensorException:
                    tensor_meta = _extract_tensor_metadata(x)
                return tensor_meta
            else:
                return None
        node.meta['tensor_meta'] = pytree.tree_map(make_tensor_meta, value)
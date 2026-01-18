import enum
import dis
import copy
import sys
import torch
import inspect
import operator
import traceback
import collections
from dataclasses import is_dataclass, fields
from .graph import magic_methods, reflectable_magic_methods, Graph
from typing import Tuple, Dict, OrderedDict, Optional, Any, Iterator, Callable
from .node import Target, Node, Argument, base_types, map_aggregate
from ._compatibility import compatibility
from .operator_schemas import check_for_mutable_operation
import torch.fx.traceback as fx_traceback
@compatibility(is_backward_compatible=True)
class TracerBase:
    graph: Graph
    record_stack_traces: bool = False
    check_mutable_operations: bool = False
    trace_asserts: bool = False
    proxy_buffer_attributes: bool = False
    traced_func_name: str = 'forward'
    scope: Scope
    module_stack: OrderedDict[str, Tuple[str, Any]]
    node_name_to_scope: Dict[str, Tuple[str, type]]

    @compatibility(is_backward_compatible=True)
    def create_node(self, kind: str, target: Target, args: Tuple[Argument, ...], kwargs: Dict[str, Argument], name: Optional[str]=None, type_expr: Optional[Any]=None) -> Node:
        """
        Inserts a graph node given target, args, kwargs, and name.

        This method can be overridden to do extra checking, validation, or
        modification of values used in node creation. For example, one might
        want to disallow in-place operations from being recorded.
        """
        if kind == 'call_function' and self.check_mutable_operations:
            check_for_mutable_operation(target, args, kwargs)
        node = self.graph.create_node(kind, target, args, kwargs, name, type_expr)
        self.node_name_to_scope[node.name] = (self.scope.module_path, self.scope.module_type)
        if fx_traceback.has_preserved_node_meta():
            current_meta: Dict[str, Any] = fx_traceback.get_current_meta()
            stack_trace = current_meta.get('stack_trace')
            if stack_trace:
                node.stack_trace = stack_trace
            for field in _COPY_META_FIELDS:
                if field in current_meta:
                    node.meta[field] = copy.copy(current_meta[field])
            new_seq_nr = torch.autograd._get_sequence_nr() - 1
            if current_meta.get('in_grad_fn', False):
                new_seq_nr = current_meta['grad_fn_seq_nr']
            node.meta['seq_nr'] = new_seq_nr
        elif self.module_stack:
            node.meta['nn_module_stack'] = copy.copy(self.module_stack)
        return node

    @compatibility(is_backward_compatible=True)
    def proxy(self, node: Node) -> 'Proxy':
        return Proxy(node, self)

    @compatibility(is_backward_compatible=True)
    def create_proxy(self, kind: str, target: Target, args: Tuple[Any, ...], kwargs: Dict[str, Any], name: Optional[str]=None, type_expr: Optional[Any]=None, proxy_factory_fn: Callable[[Node], 'Proxy']=None):
        """
        Create a Node from the given arguments, then return the Node
        wrapped in a Proxy object.

        If kind = 'placeholder', then we're creating a Node that
        represents the parameter of a function. If we need to encode
        a default parameter, we use the ``args`` tuple. ``args`` is
        otherwise empty for ``placeholder`` Nodes.
        """
        args_ = self.create_arg(args)
        kwargs_ = self.create_arg(kwargs)
        assert isinstance(args_, tuple)
        assert isinstance(kwargs_, dict)
        node = self.create_node(kind, target, args_, kwargs_, name, type_expr)
        if not proxy_factory_fn:
            proxy = self.proxy(node)
        else:
            proxy = proxy_factory_fn(node)
        if self.record_stack_traces and (not proxy.node.stack_trace):
            user_frame = self._find_user_frame()
            if user_frame:
                summary = traceback.extract_stack(user_frame)
                tb_lines = summary.format()
                proxy.node.stack_trace = ''.join(tb_lines)
        return proxy

    def _find_user_frame(self):
        """
        Find the Python stack frame executing the user code during
        symbolic tracing.
        """
        frame = inspect.currentframe()
        pt_files = ['torch/fx/proxy.py', 'torch/fx/_symbolic_trace.py', 'torch/fx/experimental/proxy_tensor.py', 'torch/_ops.py', 'torch/_tensor.py', 'torch/utils/_python_dispatch.py', 'torch/_prims_common/wrappers.py', 'torch/_refs/__init__.py', 'torch/_refs/nn/functional/__init__.py', 'torch/utils/_stats.py']
        while frame:
            frame = frame.f_back
            if frame and all((not frame.f_code.co_filename.endswith(file) for file in pt_files)):
                break
        if not frame:
            return None
        return frame

    @compatibility(is_backward_compatible=True)
    def create_arg(self, a: Any) -> Argument:
        """
        A method that lowers the objects seen as arguments during symbolic evaluation
        into Argument types that can be stored in IR.

        Can be override to support more trace-specific types.
        """
        if not isinstance(a, Proxy) and hasattr(a, '__fx_create_arg__'):
            return a.__fx_create_arg__(self)
        elif isinstance(a, tuple) and hasattr(a, '_fields'):
            args = tuple((self.create_arg(elem) for elem in a))
            return type(a)(*args)
        elif isinstance(a, (tuple, list)):
            return type(a)((self.create_arg(elem) for elem in a))
        elif isinstance(a, dict):
            r = {}
            for k, v in a.items():
                k = self.create_arg(k)

                def no_node(arg):
                    if isinstance(arg, Node):
                        raise RuntimeError(f'Keys for dictionaries used as an argument cannot contain a Node. Got key: {k}')
                map_aggregate(k, no_node)
                r[k] = self.create_arg(v)
            return r
        elif isinstance(a, slice):
            return slice(self.create_arg(a.start), self.create_arg(a.stop), self.create_arg(a.step))
        elif isinstance(a, range):
            return range(self.create_arg(a.start), self.create_arg(a.stop), self.create_arg(a.step))
        elif isinstance(a, torch._ops.OpOverload):
            return a
        if isinstance(a, Proxy):
            return a.node
        if is_dataclass(a):
            kwargs = {field.name: self.create_arg(getattr(a, field.name)) for field in fields(a)}
            return self.create_node('call_function', a.__class__, (), kwargs)
        elif isinstance(a, (*base_types, enum.Enum)) or a is None or a is ...:
            return a
        raise NotImplementedError(f'argument of type: {type(a)}')

    @compatibility(is_backward_compatible=True)
    def to_bool(self, obj: 'Proxy') -> bool:
        """Called when a proxy object is being converted to a boolean, such as
        when used in control flow.  Normally we don't know what to do because
        we don't know the value of the proxy, but a custom tracer can attach more
        information to the graph node using create_node and can choose to return a value.
        """
        raise TraceError('symbolically traced variables cannot be used as inputs to control flow')

    @compatibility(is_backward_compatible=True)
    def iter(self, obj: 'Proxy') -> Iterator:
        """Called when a proxy object is being iterated over, such as
        when used in control flow.  Normally we don't know what to do because
        we don't know the value of the proxy, but a custom tracer can attach more
        information to the graph node using create_node and can choose to return an iterator.
        """
        raise TraceError('Proxy object cannot be iterated. This can be attempted when the Proxy is used in a loop or as a *args or **kwargs function argument. See the torch.fx docs on pytorch.org for a more detailed explanation of what types of control flow can be traced, and check out the Proxy docstring for help troubleshooting Proxy iteration errors')

    @compatibility(is_backward_compatible=True)
    def keys(self, obj: 'Proxy') -> Any:
        """Called when a proxy object is has the keys() method called.
        This is what happens when ** is called on a proxy. This should return an
        iterator it ** is suppose to work in your custom tracer.
        """
        return Attribute(obj, 'keys')()
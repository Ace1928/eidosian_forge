import operator
from typing import Any, Callable, Dict, Tuple, Optional
import torch
import torch.fx
import torch.fx as fx
from torch.fx import Transformer, Proxy
from torch.fx.node import Argument, Target, Node, map_aggregate
from torch.fx.operator_schemas import (
from .schema_type_annotation import AnnotateTypesWithSchema
class NormalizeArgs(Transformer):
    """
    Normalize arguments to Python targets. This means that
    `args/kwargs` will be matched up to the module/functional's
    signature and rewritten to exclusively kwargs in positional order
    if `normalize_to_only_use_kwargs` is true. Also populates default
    values. Does not support positional-only parameters or varargs
    parameters (*args, **kwargs).

    If the nodes have 'type' metadata, it will use it to disambiguate
    overloads. Otherwise, it will throw an error.

    Example usage:
        m = torchvision.models.resnet18()
        traced = torch.fx.symbolic_trace(m)
        traced = NormalizeArgs(traced).transform()
    """

    def __init__(self, module: torch.fx.GraphModule, normalize_to_only_use_kwargs: bool=True):
        super().__init__(module)
        self.node_map: Dict[Proxy, Node] = {}
        self.normalize_to_only_use_kwargs = normalize_to_only_use_kwargs

    def run_node(self, n: Node) -> Any:
        args, kwargs = self.fetch_args_kwargs_from_env(n)

        def get_type(arg):
            if isinstance(arg, fx.Node):
                return n.meta['type'] if 'type' in n.meta else None
            return type(arg)
        arg_types = map_aggregate(n.args, get_type)
        assert isinstance(arg_types, tuple)
        arg_types = tuple([create_type_hint(i) for i in arg_types])
        kwarg_types = {k: get_type(v) for k, v in kwargs.items()}
        if n.op == 'call_function':
            out = self.call_function(n.target, args, kwargs, arg_types, kwarg_types)
        else:
            out = super().run_node(n)
        if n.op != 'output':
            self.node_map[out] = n
            out.node.meta = n.meta
            out.node.type = n.type
        return out

    def call_function(self, target: Target, args: Tuple[Argument, ...], kwargs: Dict[str, Any], arg_types: Optional[Tuple[Any, ...]]=None, kwarg_types: Optional[Dict[str, Any]]=None):
        assert callable(target)
        new_args_and_kwargs = normalize_function(target, args, kwargs, arg_types, kwarg_types, self.normalize_to_only_use_kwargs)
        if new_args_and_kwargs:
            new_args, new_kwargs = new_args_and_kwargs
            return self.tracer.create_proxy('call_function', target, new_args, new_kwargs)
        else:
            return super().call_function(target, args, kwargs)

    def call_module(self, target: Target, args: Tuple[Argument, ...], kwargs: Dict[str, Any]):
        assert isinstance(target, str)
        new_args_and_kwargs = normalize_module(self.module, target, args, kwargs, self.normalize_to_only_use_kwargs)
        if new_args_and_kwargs:
            new_args, new_kwargs = new_args_and_kwargs
            return super().call_module(target, new_args, new_kwargs)
        else:
            return super().call_module(target, args, kwargs)
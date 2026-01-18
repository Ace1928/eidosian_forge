import contextlib
import functools
import itertools
import logging
from typing import Dict, List, Optional
import torch._C
import torch.fx
import torch.nn
import torch.onnx.operators
from torch._dispatch.python import enable_python_dispatcher
from torch._dynamo.utils import deepcopy_to_fake_tensor, get_fake_value, get_real_value
from torch._dynamo.variables.base import VariableTracker
from torch._dynamo.variables.builtin import BuiltinVariable
from torch._dynamo.variables.functions import UserFunctionVariable
from torch._dynamo.variables.tensor import SymNodeVariable
from torch._guards import Source
from torch.fx.passes.shape_prop import _extract_tensor_metadata
from torch.utils import _pytree as pytree
from ..exc import (
from ..source import FSDPNNModuleSource, GetItemSource, NNModuleSource
from ..utils import proxy_args_kwargs
from .dicts import ConstDictVariable
from .lists import ListVariable, TupleVariable
from .nn_module import NNModuleVariable, UnspecializedNNModuleVariable
class WrapHigherOrderVariable(TorchHigherOrderOperatorVariable):

    def create_wrapped_node(self, tx, args, kwargs, description):
        checkpoint = tx.copy_graphstate()
        graph_checkpoint = tx.output.graph
        (body_r, treespec), body_graph, body_lifted_freevars = speculate_subgraph(tx, args[0], [*args[1:]], kwargs, graph_checkpoint, checkpoint, description, source_target=self.value, manually_set_subgraph_inputs=False, should_flatten_outputs=True)
        body_gmod = torch.fx.GraphModule(tx.output.nn_modules, body_graph)
        body_name = add_subgraph(tx, self.source, 'wrap_body', body_gmod)
        body_node = make_attr(tx, body_name)
        lifted_args = tuple((arg for arg in body_lifted_freevars.keys()))
        proxy_args = (body_node,) + lifted_args
        example_value = pytree.tree_map_only(torch.fx.Proxy, lambda a: a.node.meta['example_value'], body_r.as_proxy())
        return (proxy_args, {}, example_value, treespec, body_gmod)

    def call_function(self, tx, args: 'List[VariableTracker]', kwargs: 'Dict[str, VariableTracker]') -> 'VariableTracker':
        from .builder import wrap_fx_proxy
        p_args, p_kwargs, example_value, treespec, _ = self.create_wrapped_node(tx, args, kwargs, 'wrap')
        if len(p_kwargs) > 0:
            unimplemented('kwargs should have been flattened into lifted args')
        variable = wrap_fx_proxy(tx=tx, proxy=tx.output.create_proxy('call_function', self.value, args=tuple(p_args), kwargs={}), example_value=example_value)
        if treespec is None:
            return variable
        variable = BuiltinVariable(list).call_function(tx, [variable], {})
        return _make_inlined(tx, pytree.tree_unflatten)(variable, treespec)
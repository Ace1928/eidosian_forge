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
class MapHigherOrderVariable(TorchHigherOrderOperatorVariable):

    def call_function(self, tx, args: List[VariableTracker], kwargs: Dict[str, VariableTracker]) -> VariableTracker:
        from . import ConstantVariable, NestedUserFunctionVariable, TensorVariable, UserFunctionVariable
        from .builder import wrap_fx_proxy
        if len(kwargs) > 0:
            unimplemented('torch.ops.higher_order.map: kwargs are not supported in the map operator.')
        assert type(args[0].realize()) in (UserFunctionVariable, NestedUserFunctionVariable)
        assert type(args[1].realize()) is TensorVariable
        sample_shape = get_fake_value(args[1].as_proxy().node, tx).size()
        if len(sample_shape) < 1 or sample_shape[0] == 0:
            unimplemented("map() operator doesn't support scalar or zero-sized tensors during tracing.")
        checkpoint = tx.copy_graphstate()
        first_dim = args[1].call_method(tx, '__getitem__', args=[ConstantVariable.create(0)], kwargs={})
        (body_r, _), body_graph, body_lifted_freevars = speculate_subgraph(tx, args[0], [first_dim, *args[2:]], {}, tx.output.graph, checkpoint, 'torch.ops.higher_order.map', source_target=self.value)
        body_nn_modules = tx.copy_graphstate().output.nn_modules
        body_name = add_subgraph(tx, self.source, 'map_body', torch.fx.GraphModule(body_nn_modules.nn_modules, body_graph))
        body_node = make_attr(tx, body_name)
        p_args = (body_node, *(arg.as_proxy() for arg in args[1:]), *(arg for arg in body_lifted_freevars.keys()))
        non_single_tensor_return_unsupported('torch.ops.higher_order.map', body_r)
        r = body_r.as_proxy().node.meta['example_value']
        example_value = r.new_empty([sample_shape[0], *r.shape])
        return wrap_fx_proxy(tx=tx, proxy=tx.output.create_proxy('call_function', self.value, args=tuple(p_args), kwargs={}), example_value=example_value)
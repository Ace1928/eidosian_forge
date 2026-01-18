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
class FunctorchGradHigherOrderVariable(TorchHigherOrderOperatorVariable):

    def call_function(self, tx, args: 'List[VariableTracker]', kwargs: 'Dict[str, VariableTracker]') -> 'VariableTracker':
        from . import ConstantVariable
        from .builder import wrap_fx_proxy
        if not torch._dynamo.config.capture_func_transforms:
            unimplemented('torch.func.grad capture is disabled, it can be turned on by setting `torch._dynamo.config.capture_func_transforms=True`')
        checkpoint = tx.copy_graphstate()
        graph_checkpoint = tx.output.graph
        grad_args = (args[0], args[1], args[2])
        func, argnums, has_aux = grad_args
        kwargs = args[4].items
        if len(kwargs) > 0:
            unimplemented('torch.func.grad: kwargs arguments are currently unsupported.')
        (body_r, _), body_graph, body_lifted_freevars = speculate_subgraph(tx, func, args[3].items, {}, graph_checkpoint, checkpoint, 'torch.func.grad', source_target=self.value, enable_grad=True)
        body_name = add_subgraph(tx, self.source, 'grad_body', torch.fx.GraphModule(tx.output.nn_modules, body_graph))
        body_node = make_attr(tx, body_name)
        grad_proxy_args = (body_node, *(arg.as_proxy() for arg in grad_args[1:]))
        grad_fn = tx.output.create_proxy('call_function', torch.func.grad, args=tuple(grad_proxy_args), kwargs={}, name='grad_proxy')
        args = args[3].items
        grad_fn_args = tuple((arg.as_proxy() for arg in args)) + tuple(body_lifted_freevars)
        grad_output = grad_fn(*grad_fn_args)

        def _from_args(idx):
            return args[idx].as_proxy().node.meta['example_value'].contiguous()

        def to_python_ints(argnums):
            if not isinstance(argnums, (ConstantVariable, TupleVariable)):
                raise UserError(UserErrorType.INVALID_INPUT, f'argnums is expected to be int or tuple of ints. Got {argnums}.')
            if isinstance(argnums, ConstantVariable):
                if not isinstance(argnums.value, (int, tuple)):
                    raise UserError(UserErrorType.INVALID_INPUT, f'argnums is expected to be int or tuple of ints. Got {argnums}.')
                return argnums.value
            else:
                const_vars = argnums.unpack_var_sequence(tx)
                if not all((isinstance(var, ConstantVariable) and isinstance(var.value, int) for var in const_vars)):
                    raise UserError(UserErrorType.INVALID_INPUT, f'argnums is expected to contain int only. Got {const_vars}.')
                return tuple((var.value for var in const_vars))
        argnums_v = to_python_ints(argnums)
        example_value = pytree.tree_map(_from_args, argnums_v)
        if has_aux.value:
            body_r_proxy = body_r.as_proxy()
            aux = body_r_proxy[1].node.meta['example_value']
            example_value = (example_value, aux)
        fx_proxy = wrap_fx_proxy(tx=tx, proxy=grad_output, example_value=example_value)
        if not has_aux.value:
            if isinstance(argnums_v, int):
                return fx_proxy.call_method(tx, 'contiguous', (), {})
            else:
                grads = fx_proxy
                items = []
                for idx in range(len(argnums_v)):
                    proxy = grads.call_method(tx, '__getitem__', (ConstantVariable.create(idx),), {}).call_method(tx, 'contiguous', (), {})
                    items.append(proxy)
                return TupleVariable(items)
        else:
            grads = fx_proxy.call_method(tx, '__getitem__', (ConstantVariable.create(0),), {})
            aux = fx_proxy.call_method(tx, '__getitem__', (ConstantVariable.create(1),), {})
            if isinstance(argnums_v, int):
                return TupleVariable([grads.call_method(tx, 'contiguous', (), {}), aux])
            else:
                items = []
                for idx in range(len(argnums_v)):
                    proxy = grads.call_method(tx, '__getitem__', (ConstantVariable.create(idx),), {}).call_method(tx, 'contiguous', (), {})
                    items.append(proxy)
                return TupleVariable([TupleVariable(items), aux])
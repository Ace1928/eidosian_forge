import torch
import operator
import warnings
from typing import Callable, Dict, Iterable
from torch.fx._symbolic_trace import _assert_is_none
from torch.fx.experimental.migrate_gradual_types.constraint import ApplyBroadcasting, CalcProduct, \
from torch.fx.experimental.migrate_gradual_types.operation import \
from torch.fx.node import Target, Node
from torch.fx.experimental.migrate_gradual_types.util import gen_tensor_dims, gen_nat_constraints, gen_dvar, gen_tvar, \
from torch.fx.tensor_type import Dyn, TensorType
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.batchnorm import BatchNorm2d
class ConstraintGenerator:

    def __init__(self, traced, graph=None):
        self.traced = traced
        self.traced_params = dict(self.traced.named_parameters())
        self.constraints = []
        self.symbol_dict = {}
        self.graph = traced.graph if hasattr(traced, 'graph') else graph

    def generate_constraints(self, counter=0):
        """
        Iterate through every node and generate constraints
        Effect: self.constraints will be populated with the final constraints
        """
        graph = self.graph
        all_constraints = []
        for n in graph.nodes:
            constraints, counter = self.generate_constraints_node(n, counter)
            all_constraints += constraints
        return (Conj(all_constraints), counter)

    def generate_constraints_node(self, n: Node, counter):
        """
        Generate constraints the given node:
        Currently supported operations:
        - Reshape
        - Add
        - conv2d
        """
        if n.op == 'placeholder':
            x, counter = gen_tvar(counter)
            self.symbol_dict[n] = x
            my_type = n.type
            if n.type != Dyn and (not isinstance(n.type, TensorType)):
                if n.type == torch.nn.parameter.Parameter:
                    assert 'example_value' in n.meta
                    my_type = TensorType(n.meta['example_value'].size())
                else:
                    my_type = Dyn
            c1 = BinConstraintT(my_type, x, op_precision)
            c2 = BinConstraintT(x, MAX_TENSOR_RANK, op_leq)
            return ([c1, c2], counter)
        elif n.op == 'call_function':
            if n.target in _INFERENCE_RULES:
                return _INFERENCE_RULES[n.target](n, self.symbol_dict, self.constraints, counter)
            else:
                raise RuntimeError(f'No inference rule registered for target {n.target}!')
        elif n.op == 'call_module':
            module_instance = self.traced.get_submodule(n.target)
            if type(module_instance) in _INFERENCE_RULES:
                return _INFERENCE_RULES[type(module_instance)](n, module_instance, self.symbol_dict, self.constraints, counter)
            else:
                raise RuntimeError(f'No inference rule registered for class {type(module_instance)}!')
        elif n.op == 'call_method':
            if n.target in _INFERENCE_RULES:
                return _INFERENCE_RULES[n.target](n, self.symbol_dict, self.constraints, counter)
            else:
                raise RuntimeError(f'No inference rule registered for target {n.target}!')
        elif n.op == 'get_attr':
            t = self.traced_params.get(n.target, None)
            if isinstance(t, torch.Tensor):
                if len(t.shape) > 0:
                    res = []
                    for d in t.shape:
                        res.append(d)
                    attr_type = TensorType(res)
                    output, counter = gen_tvar(counter)
                    self.symbol_dict[n] = output
                    return ([BinConstraintT(output, attr_type, op_eq)], counter)
                else:
                    return ([], counter)
            else:
                return ([], counter)
        elif n.op == 'output':
            return ([], counter)
        else:
            raise NotImplementedError(f'Method {n.op} not yet implemented')
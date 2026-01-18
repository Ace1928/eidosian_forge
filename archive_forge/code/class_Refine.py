from functools import reduce
import torch
import operator
from torch.fx.tensor_type import Dyn, is_consistent, TensorType, is_more_precise
from typing import Callable, Dict
from torch.fx.node import Target, Node
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.conv import Conv2d
from torch.fx.experimental.refinement_types import Equality
import itertools
from torch.fx.experimental.unification import Var  # type: ignore[attr-defined]
import sympy
class Refine:
    """
    Symbolic shape inference.
    Generates constraints over type variables.
    Currently all constraints are equality constraints.
    """

    def __init__(self, traced):
        self.constraints = []
        self.traced = traced
        self.symbol_iter = itertools.count(start=0, step=1)

    def refine(self):
        """
        Generates constraints for
        every node in the graph based on
        the operation.
        """
        graph = self.traced.graph
        for n in graph.nodes:
            self.refine_node(n)
        return True

    def symbolic_relations(self):
        """
        Infers algebraic relations
        """
        graph = self.traced.graph
        for n in graph.nodes:
            self.infer_symbolic_relations(n)
        return True

    def replace_dyn_with_fresh_var(self, typ):
        """
        Replace all unknown types with fresh type variables.
        """
        if typ == Dyn:
            new_symbol = Var(next(self.symbol_iter))
            return new_symbol
        elif isinstance(typ, TensorType):
            new_args = [self.replace_dyn_with_fresh_var(a) for a in typ.__args__]
            return TensorType(tuple(new_args))
        elif isinstance(typ, list):
            return [self.replace_dyn_with_fresh_var(t) for t in typ]
        elif isinstance(typ, tuple):
            return (self.replace_dyn_with_fresh_var(t) for t in typ)
        else:
            return typ

    def convert_to_sympy_symbols(self, typ):
        """
        Replace all unknown types with fresh type variables.
        """
        if isinstance(typ, Var):
            return sympy.symbols(str(typ))
        elif isinstance(typ, TensorType):
            new_args = [self.convert_to_sympy_symbols(a) for a in typ.__args__]
            return TensorType(tuple(new_args))
        elif isinstance(typ, list):
            return [self.convert_to_sympy_symbols(t) for t in typ]
        elif isinstance(typ, tuple):
            return (self.convert_to_sympy_symbols(t) for t in typ)
        else:
            return typ

    def refine_node(self, n: Node):
        """
        Returns a list of equality constraints for
        call_module and call_function nodes.
        Models the relation between input and output dimensions
        using constraints in case they are both tensors.
        All operations used in resnet50 are defined.
        """
        if n.type is None:
            n.type = Dyn
        n.type = self.replace_dyn_with_fresh_var(n.type)
        if n.op == 'call_function':
            if n.target in _REFINEMENT_RULES:
                self.constraints += _REFINEMENT_RULES[n.target](n)
            else:
                pass
        if n.op == 'call_module':
            module_instance = self.traced.get_submodule(n.target)
            if type(module_instance) in _REFINEMENT_RULES:
                self.constraints += _REFINEMENT_RULES[type(module_instance)](n)
            else:
                pass
        if n.op == 'output':

            def get_node_type(a):
                return a.type
            n.type = torch.fx.node.map_arg(n.args[0], get_node_type)
            return n.type
        else:
            pass

    def infer_symbolic_relations(self, n: Node):
        n.type = self.convert_to_sympy_symbols(n.type)
        if n.op == 'call_function':
            if n.target in _RULES:
                return _RULES[n.target](n)
            else:
                pass
        if n.op == 'call_module':
            module_instance = self.traced.get_submodule(n.target)
            if type(module_instance) in _RULES:
                return _RULES[type(module_instance)](n, module_instance)
            else:
                pass
        if n.op == 'output':

            def get_node_type(a):
                return a.type
            n.type = torch.fx.node.map_arg(n.args[0], get_node_type)
            return n.type
        else:
            pass
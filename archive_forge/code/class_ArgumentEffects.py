from pythran.analyses.aliases import Aliases
from pythran.analyses.intrinsics import Intrinsics
from pythran.analyses.global_declarations import GlobalDeclarations
from pythran.passmanager import ModuleAnalysis
from pythran.tables import MODULES
from pythran.graph import DiGraph
from pythran import intrinsic
import gast as ast
from functools import reduce
class ArgumentEffects(ModuleAnalysis):
    """Gathers inter-procedural effects on function arguments."""

    def __init__(self):
        self.result = DiGraph()
        self.node_to_functioneffect = {}
        super(ArgumentEffects, self).__init__(Aliases, GlobalDeclarations, Intrinsics)

    def prepare(self, node):
        """
        Initialise arguments effects as this analyse is inter-procedural.

        Initialisation done for Pythonic functions and default value set for
        user defined functions.
        """
        super(ArgumentEffects, self).prepare(node)
        for i in self.intrinsics:
            fe = IntrinsicArgumentEffects[i]
            self.node_to_functioneffect[i] = fe
            self.result.add_node(fe)
        for n in self.global_declarations.values():
            fe = FunctionEffects(n)
            self.node_to_functioneffect[n] = fe
            self.result.add_node(fe)

    def run(self, node):
        result = super(ArgumentEffects, self).run(node)
        candidates = set(result)
        while candidates:
            function = candidates.pop()
            for ue in enumerate(function.update_effects):
                update_effect_idx, update_effect = ue
                if not update_effect:
                    continue
                for pred in result.successors(function):
                    edge = result.edges[function, pred]
                    for fp in enumerate(edge['formal_parameters']):
                        i, formal_parameter_idx = fp
                        ith_effectiv = edge['effective_parameters'][i]
                        if formal_parameter_idx == update_effect_idx and (not pred.update_effects[ith_effectiv]):
                            pred.update_effects[ith_effectiv] = True
                            candidates.add(pred)
        self.result = {f.func: f.update_effects for f in result}
        return self.result

    def argument_index(self, node):
        while isinstance(node, ast.Subscript):
            node = node.value
        for node_alias in self.aliases[node]:
            while isinstance(node_alias, ast.Subscript):
                node_alias = node_alias.value
            if node_alias in self.current_arguments:
                return self.current_arguments[node_alias]
            if node_alias in self.current_subscripted_arguments:
                return self.current_subscripted_arguments[node_alias]
        return -1

    def visit_FunctionDef(self, node):
        self.current_function = self.node_to_functioneffect[node]
        self.current_arguments = {arg: i for i, arg in enumerate(node.args.args)}
        self.current_subscripted_arguments = dict()
        assert self.current_function in self.result
        self.generic_visit(node)

    def visit_For(self, node):
        ai = self.argument_index(node.iter)
        if ai >= 0:
            self.current_subscripted_arguments[node.target] = ai
        self.generic_visit(node)

    def visit_AugAssign(self, node):
        n = self.argument_index(node.target)
        if n >= 0:
            self.current_function.update_effects[n] = True
        self.generic_visit(node)

    def visit_Assign(self, node):
        for t in node.targets:
            if isinstance(t, ast.Subscript):
                n = self.argument_index(t)
                if n >= 0:
                    self.current_function.update_effects[n] = True
        self.generic_visit(node)

    def visit_Call(self, node):
        for i, arg in enumerate(node.args):
            n = self.argument_index(arg)
            if n >= 0:
                func_aliases = self.aliases[node.func]
                if func_aliases is None:
                    self.current_function.update_effects[n] = True
                    continue
                func_aliases = reduce(lambda x, y: x + (list(self.node_to_functioneffect.keys()) if isinstance(y, ast.Name) and self.argument_index(y) >= 0 else [y]), func_aliases, list())
                for func_alias in func_aliases:
                    if isinstance(func_alias, ast.Call):
                        bound_name = func_alias.args[0].id
                        func_alias = self.global_declarations[bound_name]
                    if func_alias is intrinsic.UnboundValue:
                        continue
                    if func_alias not in self.node_to_functioneffect:
                        continue
                    if func_alias is MODULES['functools']['partial']:
                        base_func_aliases = self.aliases[node.args[0]]
                        fe = self.node_to_functioneffect[func_alias]
                        if len(base_func_aliases) == 1:
                            base_func_alias = next(iter(base_func_aliases))
                            fe = self.node_to_functioneffect.get(base_func_alias, fe)
                    else:
                        fe = self.node_to_functioneffect[func_alias]
                    if not self.result.has_edge(fe, self.current_function):
                        self.result.add_edge(fe, self.current_function, effective_parameters=[], formal_parameters=[])
                    edge = self.result.edges[fe, self.current_function]
                    edge['effective_parameters'].append(n)
                    edge['formal_parameters'].append(i)
        self.generic_visit(node)
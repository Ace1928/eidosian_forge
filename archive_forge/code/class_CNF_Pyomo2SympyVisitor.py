from pyomo.common import DeveloperError
from pyomo.common.collections import ComponentMap
from pyomo.common.dependencies import attempt_import
from pyomo.core.expr.logical_expr import special_boolean_atom_types
from pyomo.core.expr.numvalue import native_types, value
from pyomo.core.expr.sympy_tools import (
class CNF_Pyomo2SympyVisitor(Pyomo2SympyVisitor):

    def __init__(self, object_map, bool_varlist):
        super().__init__(object_map)
        self.boolean_variable_list = bool_varlist
        self.special_atom_map = ComponentMap()

    def beforeChild(self, node, child, child_idx):
        descend, result = super().beforeChild(node, child, child_idx)
        if descend:
            if child.__class__ in special_boolean_atom_types:
                indicator_var = self.boolean_variable_list.add()
                self.special_atom_map[indicator_var] = child
                return (False, self.object_map.getSympySymbol(indicator_var))
        return (descend, result)
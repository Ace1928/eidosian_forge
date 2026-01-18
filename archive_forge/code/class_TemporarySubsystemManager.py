from pyomo.core.base.block import Block
from pyomo.core.base.reference import Reference
from pyomo.core.expr.visitor import identify_variables
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.common.modeling import unique_component_name
from pyomo.core.base.constraint import Constraint
from pyomo.core.base.expression import Expression
from pyomo.core.base.external import ExternalFunction
from pyomo.core.expr.visitor import StreamBasedExpressionVisitor
from pyomo.core.expr.numeric_expr import ExternalFunctionExpression
from pyomo.core.expr.numvalue import native_types
class TemporarySubsystemManager(object):
    """This class is a context manager for cases when we want to
    temporarily fix or deactivate certain variables or constraints
    in order to perform some solve or calculation with the resulting
    subsystem.

    """

    def __init__(self, to_fix=None, to_deactivate=None, to_reset=None, to_unfix=None):
        """
        Arguments
        ---------
        to_fix: List
            List of var data objects that should be temporarily fixed.
            These are restored to their original status on exit from
            this object's context manager.
        to_deactivate: List
            List of constraint data objects that should be temporarily
            deactivated. These are restored to their original status on
            exit from this object's context manager.
        to_reset: List
            List of var data objects that should be reset to their
            original values on exit from this object's context context
            manager.
        to_unfix: List
            List of var data objects to be temporarily unfixed. These are
            restored to their original status on exit from this object's
            context manager.

        """
        if to_fix is None:
            to_fix = []
        if to_deactivate is None:
            to_deactivate = []
        if to_reset is None:
            to_reset = []
        if to_unfix is None:
            to_unfix = []
        if not ComponentSet(to_fix).isdisjoint(ComponentSet(to_unfix)):
            to_unfix_set = ComponentSet(to_unfix)
            both = [var for var in to_fix if var in to_unfix_set]
            var_names = '\n' + '\n'.join([var.name for var in both])
            raise RuntimeError(f'Conflicting instructions: The following variables are present in both to_fix and to_unfix lists: {{var_names}}')
        self._vars_to_fix = to_fix
        self._cons_to_deactivate = to_deactivate
        self._comps_to_set = to_reset
        self._vars_to_unfix = to_unfix
        self._var_was_fixed = None
        self._con_was_active = None
        self._comp_original_value = None
        self._var_was_unfixed = None

    def __enter__(self):
        to_fix = self._vars_to_fix
        to_deactivate = self._cons_to_deactivate
        to_set = self._comps_to_set
        to_unfix = self._vars_to_unfix
        self._var_was_fixed = [(var, var.fixed) for var in to_fix + to_unfix]
        self._con_was_active = [(con, con.active) for con in to_deactivate]
        self._comp_original_value = [(comp, comp.value) for comp in to_set]
        for var in self._vars_to_fix:
            var.fix()
        for con in self._cons_to_deactivate:
            con.deactivate()
        for var in self._vars_to_unfix:
            var.unfix()
        return self

    def __exit__(self, ex_type, ex_val, ex_bt):
        for var, was_fixed in self._var_was_fixed:
            if was_fixed:
                var.fix()
            else:
                var.unfix()
        for con, was_active in self._con_was_active:
            if was_active:
                con.activate()
        for comp, val in self._comp_original_value:
            comp.set_value(val)
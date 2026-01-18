import logging
from weakref import ref as weakref_ref, ReferenceType
from pyomo.common.deprecation import deprecation_warning, RenamedClass
from pyomo.common.log import is_debug_set
from pyomo.common.modeling import unique_component_name, NOTSET
from pyomo.common.timing import ConstructionTimer
from pyomo.core.staleflag import StaleFlagManager
from pyomo.core.expr.boolean_value import BooleanValue
from pyomo.core.expr import GetItemExpression
from pyomo.core.expr.numvalue import value
from pyomo.core.base.component import ComponentData, ModelComponentFactory
from pyomo.core.base.global_set import UnindexedComponent_index
from pyomo.core.base.indexed_component import IndexedComponent, UnindexedComponent_set
from pyomo.core.base.misc import apply_indexed_rule
from pyomo.core.base.set import Set, BooleanSet, Binary
from pyomo.core.base.util import is_functor
from pyomo.core.base.var import Var
class IndexedBooleanVar(BooleanVar):
    """An array of variables."""

    def fix(self, value=NOTSET, skip_validation=False):
        """Fix all variables in this IndexedBooleanVar (treat as nonvariable)

        This sets the `fixed` indicator to True for every variable in
        this IndexedBooleanVar.  If ``value`` is provided, the value
        (and the ``skip_validation`` flag) are first passed to
        :py:meth:`set_value()`.

        """
        for boolean_vardata in self.values():
            boolean_vardata.fix(value, skip_validation)

    def unfix(self):
        """Unfix all variables in this IndexedBooleanVar (treat as variable)

        This sets the `fixed` indicator to False for every variable in
        this IndexedBooleanVar.

        """
        for boolean_vardata in self.values():
            boolean_vardata.unfix()

    def free(self):
        """Alias for :py:meth:`unfix`"""
        return self.unfix()

    @property
    def domain(self):
        return BooleanSet

    def __getitem__(self, args):
        tmp = args if args.__class__ is tuple else (args,)
        if any((hasattr(arg, 'is_potentially_variable') and arg.is_potentially_variable() for arg in tmp)):
            return GetItemExpression((self,) + tmp)
        return super().__getitem__(args)
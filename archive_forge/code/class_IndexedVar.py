import logging
import sys
from pyomo.common.pyomo_typing import overload
from weakref import ref as weakref_ref
from pyomo.common.deprecation import RenamedClass
from pyomo.common.log import is_debug_set
from pyomo.common.modeling import NOTSET
from pyomo.common.timing import ConstructionTimer
from pyomo.core.staleflag import StaleFlagManager
from pyomo.core.expr import GetItemExpression
from pyomo.core.expr.numeric_expr import NPV_MaxExpression, NPV_MinExpression
from pyomo.core.expr.numvalue import (
from pyomo.core.base.component import ComponentData, ModelComponentFactory
from pyomo.core.base.global_set import UnindexedComponent_index
from pyomo.core.base.disable_methods import disable_methods
from pyomo.core.base.indexed_component import (
from pyomo.core.base.initializer import (
from pyomo.core.base.set import (
from pyomo.core.base.units_container import units
class IndexedVar(Var):
    """An array of variables."""

    def setlb(self, val):
        """
        Set the lower bound for this variable.
        """
        for vardata in self.values():
            vardata.lower = val

    def setub(self, val):
        """
        Set the upper bound for this variable.
        """
        for vardata in self.values():
            vardata.upper = val

    def fix(self, value=NOTSET, skip_validation=False):
        """Fix all variables in this :class:`IndexedVar` (treat as nonvariable)

        This sets the :attr:`fixed` indicator to True for every variable
        in this IndexedVar.  If ``value`` is provided, the value (and
        the ``skip_validation`` flag) are first passed to
        :meth:`set_value`.

        """
        for vardata in self.values():
            vardata.fix(value, skip_validation)

    def unfix(self):
        """Unfix all variables in this :class:`IndexedVar` (treat as variable)

        This sets the :attr:`_VarData.fixed` indicator to False for
        every variable in this :class:`IndexedVar`.

        """
        for vardata in self.values():
            vardata.unfix()

    def free(self):
        """Alias for :meth:`unfix`"""
        return self.unfix()

    @property
    def domain(self):
        raise AttributeError('The domain is not an attribute for IndexedVar. It can be set for all indices using this property setter, but must be accessed for individual variables in this container.')

    @domain.setter
    def domain(self, domain):
        """Sets the domain for all variables in this container."""
        try:
            domain_rule = SetInitializer(domain)
            if domain_rule.constant():
                domain = domain_rule(self.parent_block(), None, self)
                for vardata in self.values():
                    vardata._domain = domain
            elif domain_rule.contains_indices():
                parent = self.parent_block()
                for index in domain_rule.indices():
                    self[index]._domain = domain_rule(parent, index, self)
            else:
                parent = self.parent_block()
                for index, vardata in self.items():
                    vardata._domain = domain_rule(parent, index, self)
        except:
            logger.error('%s is not a valid domain. Variable domains must be an instance of a Pyomo Set or convertible to a Pyomo Set.' % (domain,), extra={'id': 'E2001'})
            raise

    def __getitem__(self, args):
        try:
            return super().__getitem__(args)
        except RuntimeError:
            tmp = args if args.__class__ is tuple else (args,)
            if any((hasattr(arg, 'is_potentially_variable') and arg.is_potentially_variable() for arg in tmp)):
                return GetItemExpression((self,) + tmp)
            raise
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
class _BooleanVarData(ComponentData, BooleanValue):
    """
    This class defines the data for a single variable.

    Constructor Arguments:
        component   The BooleanVar object that owns this data.
    Public Class Attributes:
        fixed       If True, then this variable is treated as a
                        fixed constant in the model.
        stale       A Boolean indicating whether the value of this variable is
                        legitimate.  This value is true if the value should
                        be considered legitimate for purposes of reporting or
                        other interrogation.
        value       The numeric value of this variable.
    """
    __slots__ = ()

    def __init__(self, component=None):
        self._component = weakref_ref(component) if component is not None else None
        self._index = NOTSET

    def is_fixed(self):
        """Returns True if this variable is fixed, otherwise returns False."""
        return self.fixed

    def is_constant(self):
        """Returns False because this is not a constant in an expression."""
        return False

    def is_variable_type(self):
        """Returns True because this is a variable."""
        return True

    def is_potentially_variable(self):
        """Returns True because this is a variable."""
        return True

    def set_value(self, val, skip_validation=False):
        """
        Set the value of this numeric object, after
        validating its value. If the 'valid' flag is True,
        then the validation step is skipped.
        """
        if val.__class__ not in _logical_var_types:
            if not skip_validation:
                logger.warning("implicitly casting '%s' value %s to bool" % (self.name, val))
            val = bool(val)
        self._value = val
        self._stale = StaleFlagManager.get_flag(self._stale)

    def clear(self):
        self.value = None

    def __call__(self, exception=True):
        """Compute the value of this variable."""
        return self.value

    @property
    def value(self):
        """Return the value for this variable."""
        raise NotImplementedError

    @property
    def domain(self):
        """Return the domain for this variable."""
        raise NotImplementedError

    @property
    def fixed(self):
        """Return the fixed indicator for this variable."""
        raise NotImplementedError

    @property
    def stale(self):
        """Return the stale indicator for this variable."""
        raise NotImplementedError

    def fix(self, value=NOTSET, skip_validation=False):
        """Fix the value of this variable (treat as nonvariable)

        This sets the `fixed` indicator to True.  If ``value`` is
        provided, the value (and the ``skip_validation`` flag) are first
        passed to :py:meth:`set_value()`.

        """
        self.fixed = True
        if value is not NOTSET:
            self.set_value(value, skip_validation)

    def unfix(self):
        """Unfix this variable (treat as variable)

        This sets the `fixed` indicator to False.

        """
        self.fixed = False

    def free(self):
        """Alias for :py:meth:`unfix`"""
        return self.unfix()
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
class _GeneralBooleanVarData(_BooleanVarData):
    """
    This class defines the data for a single Boolean variable.

    Constructor Arguments:
        component   The BooleanVar object that owns this data.

    Public Class Attributes:
        domain      The domain of this variable.
        fixed       If True, then this variable is treated as a
                        fixed constant in the model.
        stale       A Boolean indicating whether the value of this variable is
                        legitimiate.  This value is true if the value should
                        be considered legitimate for purposes of reporting or
                        other interrogation.
        value       The numeric value of this variable.

    The domain attribute is a property because it is
    too widely accessed directly to enforce explicit getter/setter
    methods and we need to deter directly modifying or accessing
    these attributes in certain cases.
    """
    __slots__ = ('_value', 'fixed', '_stale', '_associated_binary')
    __autoslot_mappers__ = {'_associated_binary': _associated_binary_mapper, '_stale': StaleFlagManager.stale_mapper}

    def __init__(self, component=None):
        self._component = weakref_ref(component) if component is not None else None
        self._index = NOTSET
        self._value = None
        self.fixed = False
        self._stale = 0
        self._associated_binary = None

    @property
    def value(self):
        """Return (or set) the value for this variable."""
        return self._value

    @value.setter
    def value(self, val):
        self.set_value(val)

    @property
    def domain(self):
        """Return the domain for this variable."""
        return BooleanSet

    @property
    def stale(self):
        return StaleFlagManager.is_stale(self._stale)

    @stale.setter
    def stale(self, val):
        if val:
            self._stale = 0
        else:
            self._stale = StaleFlagManager.get_flag(0)

    def get_associated_binary(self):
        """Get the binary _VarData associated with this
        _GeneralBooleanVarData"""
        return self._associated_binary() if self._associated_binary is not None else None

    def associate_binary_var(self, binary_var):
        """Associate a binary _VarData to this _GeneralBooleanVarData"""
        if self._associated_binary is not None and type(self._associated_binary) is not _DeprecatedImplicitAssociatedBinaryVariable:
            raise RuntimeError("Reassociating BooleanVar '%s' (currently associated with '%s') with '%s' is not allowed" % (self.name, self._associated_binary().name if self._associated_binary is not None else None, binary_var.name if binary_var is not None else None))
        if binary_var is not None:
            self._associated_binary = weakref_ref(binary_var)
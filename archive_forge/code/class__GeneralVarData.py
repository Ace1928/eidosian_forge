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
class _GeneralVarData(_VarData):
    """This class defines the data for a single variable."""
    __slots__ = ('_value', '_lb', '_ub', '_domain', '_fixed', '_stale')
    __autoslot_mappers__ = {'_stale': StaleFlagManager.stale_mapper}

    def __init__(self, component=None):
        self._component = weakref_ref(component) if component is not None else None
        self._index = NOTSET
        self._value = None
        self._lb = None
        self._ub = None
        self._domain = None
        self._fixed = False
        self._stale = 0

    @classmethod
    def copy(cls, src):
        self = cls.__new__(cls)
        self._component = src._component
        self._value = src._value
        self._lb = src._lb
        self._ub = src._ub
        self._domain = src._domain
        self._fixed = src._fixed
        self._stale = src._stale
        self._index = src._index
        return self

    def set_value(self, val, skip_validation=False):
        """Set the current variable value.

        Set the value of this variable.  The incoming value is converted
        to a numeric value (i.e., expressions are evaluated).  If the
        variable has units, the incoming value is converted to the
        correct units before storing the value.  The final value is
        checked against both the variable domain and bounds, and an
        exception is raised if the value is not valid.  Domain and
        bounds checking can be bypassed by setting the ``skip_validation``
        argument to :const:`True`.

        """
        if val is None:
            self._value = None
            self._stale = 0
            return
        if val.__class__ in native_numeric_types:
            pass
        elif self.parent_component()._units is not None:
            _src_magnitude = value(val)
            if val.__class__ in native_numeric_types:
                val = _src_magnitude
            else:
                _src_units = units.get_units(val)
                val = units.convert_value(num_value=_src_magnitude, from_units=_src_units, to_units=self.parent_component()._units)
        else:
            val = value(val)
        if not skip_validation:
            if val not in self.domain:
                logger.warning("Setting Var '%s' to a value `%s` (%s) not in domain %s." % (self.name, val, type(val).__name__, self.domain), extra={'id': 'W1001'})
            elif self._lb is not None and val < value(self._lb) or (self._ub is not None and val > value(self._ub)):
                logger.warning("Setting Var '%s' to a numeric value `%s` outside the bounds %s." % (self.name, val, self.bounds), extra={'id': 'W1002'})
        self._value = val
        self._stale = StaleFlagManager.get_flag(self._stale)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, val):
        self.set_value(val)

    @property
    def domain(self):
        return self._domain

    @domain.setter
    def domain(self, domain):
        try:
            self._domain = SetInitializer(domain)(self.parent_block(), self.index(), self)
        except:
            logger.error('%s is not a valid domain. Variable domains must be an instance of a Pyomo Set or convertible to a Pyomo Set.' % (domain,), extra={'id': 'E2001'})
            raise

    @_VarData.bounds.getter
    def bounds(self):
        domain_lb, domain_ub = self.domain.bounds()
        lb = self._lb
        if lb.__class__ not in native_numeric_types:
            if lb is not None:
                lb = float(value(lb))
        if lb in _nonfinite_values or lb != lb:
            if lb == _ninf:
                lb = None
            else:
                raise ValueError("Var '%s' created with an invalid non-finite lower bound (%s)." % (self.name, lb))
        if domain_lb is not None:
            if lb is None:
                lb = domain_lb
            else:
                lb = max(lb, domain_lb)
        ub = self._ub
        if ub.__class__ not in native_numeric_types:
            if ub is not None:
                ub = float(value(ub))
        if ub in _nonfinite_values or ub != ub:
            if ub == _inf:
                ub = None
            else:
                raise ValueError("Var '%s' created with an invalid non-finite upper bound (%s)." % (self.name, ub))
        if domain_ub is not None:
            if ub is None:
                ub = domain_ub
            else:
                ub = min(ub, domain_ub)
        return (lb, ub)

    @_VarData.lb.getter
    def lb(self):
        domain_lb, domain_ub = self.domain.bounds()
        lb = self._lb
        if lb.__class__ not in native_numeric_types:
            if lb is not None:
                lb = float(value(lb))
        if lb in _nonfinite_values or lb != lb:
            if lb == _ninf:
                lb = None
            else:
                raise ValueError("Var '%s' created with an invalid non-finite lower bound (%s)." % (self.name, lb))
        if domain_lb is not None:
            if lb is None:
                lb = domain_lb
            else:
                lb = max(lb, domain_lb)
        return lb

    @_VarData.ub.getter
    def ub(self):
        domain_lb, domain_ub = self.domain.bounds()
        ub = self._ub
        if ub.__class__ not in native_numeric_types:
            if ub is not None:
                ub = float(value(ub))
        if ub in _nonfinite_values or ub != ub:
            if ub == _inf:
                ub = None
            else:
                raise ValueError("Var '%s' created with an invalid non-finite upper bound (%s)." % (self.name, ub))
        if domain_ub is not None:
            if ub is None:
                ub = domain_ub
            else:
                ub = min(ub, domain_ub)
        return ub

    @property
    def lower(self):
        """Return (or set) an expression for the variable lower bound.

        This returns a (not potentially variable) expression for the
        variable lower bound.  This represents the tighter of the
        current domain and the constant or expression assigned to
        :attr:`lower`.  Note that the expression will NOT automatically
        reflect changes to either the domain or the bound expression
        (e.g., because of assignment to either :attr:`lower` or
        :attr:`domain`).

        """
        dlb, _ = self.domain.bounds()
        if self._lb is None:
            return dlb
        elif dlb is None:
            return self._lb
        return NPV_MaxExpression((self._lb, dlb))

    @lower.setter
    def lower(self, val):
        self._lb = self._process_bound(val, 'lower')

    @property
    def upper(self):
        """Return (or set) an expression for the variable upper bound.

        This returns a (not potentially variable) expression for the
        variable upper bound.  This represents the tighter of the
        current domain and the constant or expression assigned to
        :attr:`upper`.  Note that the expression will NOT automatically
        reflect changes to either the domain or the bound expression
        (e.g., because of assignment to either :attr:`upper` or
        :attr:`domain`).

        """
        _, dub = self.domain.bounds()
        if self._ub is None:
            return dub
        elif dub is None:
            return self._ub
        return NPV_MinExpression((self._ub, dub))

    @upper.setter
    def upper(self, val):
        self._ub = self._process_bound(val, 'upper')

    def get_units(self):
        """Return the units for this variable entry."""
        return self.parent_component()._units

    @property
    def fixed(self):
        return self._fixed

    @fixed.setter
    def fixed(self, val):
        self._fixed = bool(val)

    @property
    def stale(self):
        return StaleFlagManager.is_stale(self._stale)

    @stale.setter
    def stale(self, val):
        if val:
            self._stale = 0
        else:
            self._stale = StaleFlagManager.get_flag(0)

    def is_fixed(self):
        return self._fixed

    def _process_bound(self, val, bound_type):
        if type(val) in native_numeric_types or val is None:
            pass
        elif is_potentially_variable(val):
            raise ValueError("Potentially variable input of type '%s' supplied as %s bound for variable '%s' - legal types must be constants or non-potentially variable expressions." % (type(val).__name__, bound_type, self.name))
        else:
            _units = self.parent_component()._units
            if _units is not None:
                val = units.convert(val, to_units=_units)
        return val
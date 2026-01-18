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
class _VarData(ComponentData, NumericValue):
    """This class defines the abstract interface for a single variable.

    Note that this "abstract" class is not intended to be directly
    instantiated.

    """
    __slots__ = ()

    def has_lb(self):
        """Returns :const:`False` when the lower bound is
        :const:`None` or negative infinity"""
        return self.lb is not None

    def has_ub(self):
        """Returns :const:`False` when the upper bound is
        :const:`None` or positive infinity"""
        return self.ub is not None

    def setlb(self, val):
        """
        Set the lower bound for this variable after validating that
        the value is fixed (or None).
        """
        self.lower = val

    def setub(self, val):
        """
        Set the upper bound for this variable after validating that
        the value is fixed (or None).
        """
        self.upper = val

    @property
    def bounds(self):
        """Returns (or set) the tuple (lower bound, upper bound).

        This returns the current (numeric) values of the lower and upper
        bounds as a tuple.  If there is no bound, returns None (and not
        +/-inf)

        """
        return (self.lb, self.ub)

    @bounds.setter
    def bounds(self, val):
        self.lower, self.upper = val

    @property
    def lb(self):
        """Return (or set) the numeric value of the variable lower bound."""
        lb = value(self.lower)
        return None if lb == _ninf else lb

    @lb.setter
    def lb(self, val):
        self.lower = val

    @property
    def ub(self):
        """Return (or set) the numeric value of the variable upper bound."""
        ub = value(self.upper)
        return None if ub == _inf else ub

    @ub.setter
    def ub(self, val):
        self.upper = val

    def is_integer(self):
        """Returns True when the domain is a contiguous integer range."""
        _id = id(self.domain)
        if _id in _known_global_real_domains:
            return not _known_global_real_domains[_id]
        _interval = self.domain.get_interval()
        if _interval is None:
            return False
        start, stop, step = _interval
        return step == 1 and (start is None or int(start) == start) and (stop is None or int(stop) == stop)

    def is_binary(self):
        """Returns True when the domain is restricted to Binary values."""
        domain = self.domain
        if domain is Binary:
            return True
        if id(domain) in _known_global_real_domains:
            return False
        return domain.get_interval() == (0, 1, 1)

    def is_continuous(self):
        """Returns True when the domain is a continuous real range"""
        _id = id(self.domain)
        if _id in _known_global_real_domains:
            return _known_global_real_domains[_id]
        _interval = self.domain.get_interval()
        return _interval is not None and _interval[2] == 0

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

    def _compute_polynomial_degree(self, result):
        """
        If the variable is fixed, it represents a constant
        is a polynomial with degree 0. Otherwise, it has
        degree 1. This method is used in expressions to
        compute polynomial degree.
        """
        if self.fixed:
            return 0
        return 1

    def clear(self):
        self.value = None

    def __call__(self, exception=True):
        """Compute the value of this variable."""
        return self.value

    def set_value(self, val, skip_validation=False):
        """Set the current variable value."""
        raise NotImplementedError

    @property
    def value(self):
        """Return (or set) the value for this variable."""
        raise NotImplementedError

    @property
    def domain(self):
        """Return (or set) the domain for this variable."""
        raise NotImplementedError

    @property
    def lower(self):
        """Return (or set) an expression for the variable lower bound."""
        raise NotImplementedError

    @property
    def upper(self):
        """Return (or set) an expression for the variable upper bound."""
        raise NotImplementedError

    @property
    def fixed(self):
        """Return (or set) the fixed indicator for this variable.

        Alias for :meth:`is_fixed` / :meth:`fix` / :meth:`unfix`.

        """
        raise NotImplementedError

    @property
    def stale(self):
        """The stale status for this variable.

        Variables are "stale" if their current value was not updated as
        part of the most recent model update.  A "model update" can be
        one of several things: a solver invocation, loading a previous
        solution, or manually updating a non-stale :class:`Var` value.

        Returns
        -------
        bool

        Notes
        -----
        Fixed :class:`Var` objects will be stale after invoking a solver
        (as their value was not updated by the solver).

        Updating a stale :class:`Var` value will not cause other
        variable values to be come stale.  However, updating the first
        non-stale :class:`Var` value after a solve or solution load
        *will* cause all other variables to be marked as stale

        """
        raise NotImplementedError

    def fix(self, value=NOTSET, skip_validation=False):
        """Fix the value of this variable (treat as nonvariable)

        This sets the :attr:`fixed` indicator to True.  If ``value`` is
        provided, the value (and the ``skip_validation`` flag) are first
        passed to :meth:`set_value()`.

        """
        self.fixed = True
        if value is not NOTSET:
            self.set_value(value, skip_validation)

    def unfix(self):
        """Unfix this variable (treat as variable in solver interfaces)

        This sets the :attr:`fixed` indicator to False.

        """
        self.fixed = False

    def free(self):
        """Alias for :meth:`unfix`"""
        return self.unfix()
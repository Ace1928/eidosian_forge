from pyomo.common.modeling import NoArgumentGiven
from pyomo.core.staleflag import StaleFlagManager
from pyomo.core.expr.numvalue import NumericValue, is_numeric_data, value
from pyomo.core.kernel.base import ICategorizedObject, _abstract_readwrite_property
from pyomo.core.kernel.container_utils import define_simple_containers
from pyomo.core.kernel.set_types import RealSet, IntegerSet
class variable(IVariable):
    """A decision variable

    Decision variables are used in objectives and
    constraints to define an optimization problem.

    Args:
        domain_type: Sets the domain type of the
            variable. Must be one of :const:`RealSet` or
            :const:`IntegerSet`. Can be updated later by
            assigning to the :attr:`domain_type`
            property. The default value of :const:`None` is
            equivalent to :const:`RealSet`, unless the
            :attr:`domain` keyword is used.
        domain: Sets the domain of the variable. This
            updates the :attr:`domain_type`, :attr:`lb`, and
            :attr:`ub` properties of the variable. The
            default value of :const:`None` implies that this
            keyword is ignored. This keyword can not be used
            in combination with the :attr:`domain_type`
            keyword.
        lb: Sets the lower bound of the variable. Can be
            updated later by assigning to the :attr:`lb`
            property on the variable. Default is
            :const:`None`, which is equivalent to
            :const:`-inf`.
        ub: Sets the upper bound of the variable. Can be
            updated later by assigning to the :attr:`ub`
            property on the variable. Default is
            :const:`None`, which is equivalent to
            :const:`+inf`.
        value: Sets the value of the variable. Can be
            updated later by assigning to the :attr:`value`
            property on the variable. Default is
            :const:`None`.
        fixed (bool): Sets the fixed status of the
            variable. Can be updated later by assigning to
            the :attr:`fixed` property or by calling the
            :meth:`fix` method. Default is :const:`False`.

    Examples:
        >>> import pyomo.kernel as pmo
        >>> # A continuous variable with infinite bounds
        >>> x = pmo.variable()
        >>> # A binary variable
        >>> x = pmo.variable(domain=pmo.Binary)
        >>> # Also a binary variable
        >>> x = pmo.variable(domain_type=pmo.IntegerSet, lb=0, ub=1)
    """
    _ctype = IVariable
    __slots__ = ('_parent', '_storage_key', '_domain_type', '_active', '_lb', '_ub', '_value', '_fixed', '_stale', '__weakref__')

    def __init__(self, domain_type=None, domain=None, lb=None, ub=None, value=None, fixed=False):
        self._parent = None
        self._storage_key = None
        self._active = True
        self._domain_type = RealSet
        self._lb = lb
        self._ub = ub
        self._value = value
        self._fixed = fixed
        self._stale = 0
        if domain_type is not None or domain is not None:
            self._domain_type, self._lb, self._ub = _extract_domain_type_and_bounds(domain_type, domain, lb, ub)

    @property
    def lower(self):
        """The lower bound of the variable"""
        return self._lb

    @lower.setter
    def lower(self, lb):
        if lb is not None and (not is_numeric_data(lb)):
            raise ValueError('Variable lower bounds must be numbers or expressions restricted to numeric data.')
        self._lb = lb

    @property
    def upper(self):
        """The upper bound of the variable"""
        return self._ub

    @upper.setter
    def upper(self, ub):
        if ub is not None and (not is_numeric_data(ub)):
            raise ValueError('Variable upper bounds must be numbers or expressions restricted to numeric data.')
        self._ub = ub

    @property
    def value(self):
        """The value of the variable"""
        return self._value

    @value.setter
    def value(self, value):
        self._value = value
        self._stale = StaleFlagManager.get_flag(self._stale)

    def set_value(self, value, skip_validation=True):
        self.value = value

    @property
    def fixed(self):
        """The fixed status of the variable"""
        return self._fixed

    @fixed.setter
    def fixed(self, fixed):
        self._fixed = fixed

    @property
    def stale(self):
        """The stale status of the variable"""
        return StaleFlagManager.is_stale(self._stale)

    @stale.setter
    def stale(self, stale):
        if stale:
            self._stale = 0
        else:
            self._stale = StaleFlagManager.get_flag(0)

    @property
    def domain_type(self):
        """The domain type of the variable (:class:`RealSet`
        or :class:`IntegerSet`)"""
        return self._domain_type

    @domain_type.setter
    def domain_type(self, domain_type):
        if domain_type not in IVariable._valid_domain_types:
            raise ValueError("Domain type '%s' is not valid. Must be one of: %s" % (self.domain_type, IVariable._valid_domain_types))
        self._domain_type = domain_type

    def _set_domain(self, domain):
        """Set the domain of the variable. This method
        updates the :attr:`domain_type` property and
        overwrites the :attr:`lb` and :attr:`ub` properties
        with the domain bounds."""
        self.domain_type, self.lb, self.ub = _extract_domain_type_and_bounds(None, domain, None, None)
    domain = property(fset=_set_domain, doc=_set_domain.__doc__)
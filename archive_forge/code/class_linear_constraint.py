from pyomo.core.expr.numvalue import (
from pyomo.core.expr.expr_common import ExpressionType
from pyomo.core.expr.relational_expr import (
from pyomo.core.kernel.base import ICategorizedObject, _abstract_readonly_property
from pyomo.core.kernel.container_utils import define_simple_containers
class linear_constraint(_MutableBoundsConstraintMixin, IConstraint):
    """A linear constraint

    A linear constraint stores a linear relational
    expression defined by a list of variables and
    coefficients. This class can be used to reduce build
    time and memory for an optimization model. It also
    increases the speed at which the model can be output to
    a solver.

    Args:
        variables (list): Sets the list of variables in the
            linear expression defining the body of the
            constraint. Can be updated later by assigning to
            the :attr:`variables` property on the
            constraint.
        coefficients (list): Sets the list of coefficients
            for the variables in the linear expression
            defining the body of the constraint. Can be
            updated later by assigning to the
            :attr:`coefficients` property on the constraint.
        terms (list): An alternative way of initializing the
            :attr:`variables` and :attr:`coefficients` lists
            using an iterable of (variable, coefficient)
            tuples. Can be updated later by assigning to the
            :attr:`terms` property on the constraint. This
            keyword should not be used in combination with
            the :attr:`variables` or :attr:`coefficients`
            keywords.
        lb: Sets the lower bound of the constraint. Can be
            updated later by assigning to the :attr:`lb`
            property on the constraint. Default is
            :const:`None`, which is equivalent to
            :const:`-inf`.
        ub: Sets the upper bound of the constraint. Can be
            updated later by assigning to the :attr:`ub`
            property on the constraint. Default is
            :const:`None`, which is equivalent to
            :const:`+inf`.
        rhs: Sets the right-hand side of the constraint. Can
            be updated later by assigning to the :attr:`rhs`
            property on the constraint. The default value of
            :const:`None` implies that this keyword is
            ignored. Otherwise, use of this keyword implies
            that the :attr:`equality` property is set to
            :const:`True`.

    Examples:
        >>> import pyomo.kernel as pmo
        >>> # Decision variables used to define constraints
        >>> x = pmo.variable()
        >>> y = pmo.variable()
        >>> # An upper bound constraint
        >>> c = pmo.linear_constraint(variables=[x,y], coefficients=[1,2], ub=1)
        >>> # (equivalent form)
        >>> c = pmo.linear_constraint(terms=[(x,1), (y,2)], ub=1)
        >>> # (equivalent form using a general constraint)
        >>> c = pmo.constraint(x + 2*y <= 1)
    """
    _ctype = IConstraint
    _linear_canonical_form = True
    __slots__ = ('_parent', '_storage_key', '_active', '_variables', '_coefficients', '_lb', '_ub', '_equality', '__weakref__')

    def __init__(self, variables=None, coefficients=None, terms=None, lb=None, ub=None, rhs=None):
        self._parent = None
        self._storage_key = None
        self._active = True
        self._variables = None
        self._coefficients = None
        self._lb = None
        self._ub = None
        self._equality = False
        if terms is not None:
            if variables is not None or coefficients is not None:
                raise ValueError("Both the 'variables' and 'coefficients' keywords must be None when the 'terms' keyword is not None")
            self.terms = terms
        elif variables is not None or coefficients is not None:
            if variables is None or coefficients is None:
                raise ValueError("Both the 'variables' and 'coefficients' keywords must be set when the 'terms' keyword is None")
            self._variables = tuple(variables)
            self._coefficients = tuple(coefficients)
        else:
            self._variables = ()
            self._coefficients = ()
        if rhs is None:
            self.lb = lb
            self.ub = ub
        else:
            if lb is not None or ub is not None:
                raise ValueError("The 'rhs' keyword can not be used with the 'lb' or 'ub' keywords to initialize a constraint.")
            self.rhs = rhs

    @property
    def terms(self):
        """An iterator over the terms in the body of this
        constraint as (variable, coefficient) tuples"""
        return zip(self._variables, self._coefficients)

    @terms.setter
    def terms(self, terms):
        """Set the terms in the body of this constraint
        using an iterable of (variable, coefficient) tuples"""
        transpose = tuple(zip(*terms))
        if len(transpose) == 2:
            self._variables, self._coefficients = transpose
        else:
            assert transpose == ()
            self._variables = ()
            self._coefficients = ()

    def __call__(self, exception=True):
        try:
            return sum((value(c, exception=exception) * v(exception=exception) for v, c in self.terms))
        except (ValueError, TypeError):
            if exception:
                raise ValueError('one or more terms could not be evaluated')
            return None

    @property
    def body(self):
        """The body of the constraint"""
        return sum((c * v for v, c in self.terms))

    def canonical_form(self, compute_values=True):
        """Build a canonical representation of the body of
        this constraints"""
        from pyomo.repn.standard_repn import StandardRepn
        variables = []
        coefficients = []
        constant = 0
        for v, c in self.terms:
            if v.is_expression_type():
                v = v.expr
            if not v.fixed:
                variables.append(v)
                if compute_values:
                    coefficients.append(value(c))
                else:
                    coefficients.append(c)
            elif compute_values:
                constant += value(c) * v()
            else:
                constant += c * v
        repn = StandardRepn()
        repn.linear_vars = tuple(variables)
        repn.linear_coefs = tuple(coefficients)
        repn.constant = constant
        return repn
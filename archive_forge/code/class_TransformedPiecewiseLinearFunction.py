import logging
import bisect
from pyomo.core.expr.numvalue import value as _value
from pyomo.core.kernel.set_types import IntegerSet
from pyomo.core.kernel.block import block
from pyomo.core.kernel.expression import expression, expression_tuple
from pyomo.core.kernel.variable import (
from pyomo.core.kernel.constraint import (
from pyomo.core.kernel.sos import sos2
from pyomo.core.kernel.piecewise_library.util import (
class TransformedPiecewiseLinearFunction(block):
    """Base class for transformed piecewise linear functions

    A transformed piecewise linear functions is a block of
    variables and constraints that enforce a piecewise
    linear relationship between an input variable and an
    output variable.

    Args:
        f (:class:`PiecewiseLinearFunction`): The piecewise
            linear function to transform.
        input: The variable constrained to be the input of
            the piecewise linear function.
        output: The variable constrained to be the output of
            the piecewise linear function.
        bound (str): The type of bound to impose on the
            output expression. Can be one of:

              - 'lb': y <= f(x)
              - 'eq': y  = f(x)
              - 'ub': y >= f(x)
        validate (bool): Indicates whether or not to perform
            validation of the input data. The default is
            :const:`True`. Validation can be performed
            manually after the piecewise object is created
            by calling the :meth:`validate`
            method. Validation should be performed any time
            the inputs are changed (e.g., when using mutable
            parameters in the breakpoints list or when the
            input variable changes).
        **kwds: Additional keywords are passed to the
            :meth:`validate` method when the :attr:`validate`
            keyword is :const:`True`; otherwise, they are
            ignored.
    """

    def __init__(self, f, input=None, output=None, bound='eq', validate=True, **kwds):
        super(TransformedPiecewiseLinearFunction, self).__init__()
        assert isinstance(f, PiecewiseLinearFunction)
        if bound not in ('lb', 'ub', 'eq'):
            raise ValueError("Invalid bound type %r. Must be one of: ['lb','ub','eq']" % bound)
        self._bound = bound
        self._f = f
        self._inout = expression_tuple([expression(input), expression(output)])
        if validate:
            self.validate(**kwds)

    @property
    def input(self):
        """The expression that stores the input to the
        piecewise function. The returned object can be
        updated by assigning to its :attr:`expr`
        attribute."""
        return self._inout[0]

    @property
    def output(self):
        """The expression that stores the output of the
        piecewise function. The returned object can be
        updated by assigning to its :attr:`expr`
        attribute."""
        return self._inout[1]

    @property
    def bound(self):
        """The bound type assigned to the piecewise
        relationship ('lb','ub','eq')."""
        return self._bound

    def validate(self, equal_slopes_tolerance=1e-06, require_bounded_input_variable=True, require_variable_domain_coverage=True):
        """
        Validate this piecewise linear function by verifying
        various properties of the breakpoints, values, and
        input variable (e.g., that the list of breakpoints
        is nondecreasing).

        Args:
            equal_slopes_tolerance (float): Tolerance used
                check if consecutive slopes are nearly
                equal. If any are found, validation will
                fail. Default is 1e-6.
            require_bounded_input_variable (bool): Indicates
                if the input variable is required to have
                finite upper and lower bounds. Default is
                :const:`True`. Setting this keyword to
                :const:`False` can be used to allow general
                expressions to be used as the input in place
                of a variable.
            require_variable_domain_coverage (bool):
                Indicates if the function domain (defined by
                the endpoints of the breakpoints list) needs
                to cover the entire domain of the input
                variable. Default is :const:`True`. Ignored
                for any bounds of variables that are not
                finite, or when the input is not assigned a
                variable.

        Returns:
            int:
                a function characterization code (see
                :func:`util.characterize_function`)

        Raises:
            PiecewiseValidationError: if validation fails
        """
        ftype = self._f.validate(equal_slopes_tolerance=equal_slopes_tolerance)
        assert ftype in (1, 2, 3, 4, 5)
        input_var = self.input.expr
        if not isinstance(input_var, IVariable):
            input_var = None
        if require_bounded_input_variable and (input_var is None or not input_var.has_lb() or (not input_var.has_ub())):
            raise PiecewiseValidationError("Piecewise function input is not a variable with finite upper and lower bounds: %s. To avoid this error, set the 'require_bounded_input_variable' keyword to False or disable validation." % str(input_var))
        if require_variable_domain_coverage and input_var is not None:
            domain_lb = _value(self.breakpoints[0])
            domain_ub = _value(self.breakpoints[-1])
            if input_var.has_lb() and _value(input_var.lb) < domain_lb:
                raise PiecewiseValidationError("Piecewise function domain does not include the lower bound of the input variable: %s.ub = %s > %s. To avoid this error, set the 'require_variable_domain_coverage' keyword to False or disable validation." % (input_var.name, _value(input_var.lb), domain_lb))
            if input_var.has_ub() and _value(input_var.ub) > domain_ub:
                raise PiecewiseValidationError("Piecewise function domain does not include the upper bound of the input variable: %s.ub = %s > %s. To avoid this error, set the 'require_variable_domain_coverage' keyword to False or disable validation." % (input_var.name, _value(input_var.ub), domain_ub))
        return ftype

    @property
    def breakpoints(self):
        """The set of breakpoints used to defined this function"""
        return self._f.breakpoints

    @property
    def values(self):
        """The set of values used to defined this function"""
        return self._f.values

    def __call__(self, x):
        """Evaluates the piecewise linear function at the
        given point using interpolation"""
        return self._f(x)
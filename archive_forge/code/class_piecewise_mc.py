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
class piecewise_mc(TransformedPiecewiseLinearFunction):
    """Discrete MC piecewise representation

    Expresses a piecewise linear function using
    the MC formulation.
    """

    def __init__(self, *args, **kwds):
        super(piecewise_mc, self).__init__(*args, **kwds)
        polytopes = range(len(self.breakpoints) - 1)
        slopes = tuple(((self.values[p + 1] - self.values[p]) / (self.breakpoints[p + 1] - self.breakpoints[p]) for p in polytopes))
        intercepts = tuple((self.values[p] - slopes[p] * self.breakpoints[p] for p in polytopes))
        self.v = variable_dict()
        lmbda = self.v['lambda'] = variable_tuple((variable() for p in polytopes))
        lmbda_tuple = tuple(lmbda)
        y = self.v['y'] = variable_tuple((variable(domain_type=IntegerSet, lb=0, ub=1) for p in polytopes))
        y_tuple = tuple(y)
        self.c = constraint_list()
        self.c.append(linear_constraint(variables=lmbda_tuple + (self.input,), coefficients=(1,) * len(lmbda) + (-1,), rhs=0))
        self.c.append(linear_constraint(variables=lmbda_tuple + y_tuple + (self.output,), coefficients=slopes + intercepts + (-1,)))
        if self.bound == 'ub':
            self.c[-1].lb = 0
        elif self.bound == 'lb':
            self.c[-1].ub = 0
        else:
            assert self.bound == 'eq'
            self.c[-1].rhs = 0
        clist1 = []
        clist2 = []
        for p in polytopes:
            clist1.append(linear_constraint(variables=(y[p], lmbda[p]), coefficients=(self.breakpoints[p], -1), ub=0))
            clist2.append(linear_constraint(variables=(lmbda[p], y[p]), coefficients=(1, -self.breakpoints[p + 1]), ub=0))
        self.c.append(constraint_tuple(clist1))
        self.c.append(constraint_tuple(clist2))
        self.c.append(linear_constraint(variables=y_tuple, coefficients=(1,) * len(y), rhs=1))

    def validate(self, **kwds):
        """
        Validate this piecewise linear function by verifying
        various properties of the breakpoints, values, and
        input variable (e.g., that the list of breakpoints
        is nondecreasing).

        See base class documentation for keyword
        descriptions.
        """
        ftype = super(piecewise_mc, self).validate(**kwds)
        if ftype == characterize_function.step:
            raise PiecewiseValidationError("The 'mc' piecewise representation does not support step functions.")
        return ftype
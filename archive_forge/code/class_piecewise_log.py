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
class piecewise_log(TransformedPiecewiseLinearFunction):
    """Discrete LOG piecewise representation

    Expresses a piecewise linear function using the LOG
    formulation. This formulation uses logarithmic number of
    discrete variables in terms of number of breakpoints.
    """

    def __init__(self, *args, **kwds):
        super(piecewise_log, self).__init__(*args, **kwds)
        breakpoints = self.breakpoints
        values = self.values
        if not is_positive_power_of_two(len(breakpoints) - 1):
            raise ValueError('The list of breakpoints must be of length (2^n)+1 for some positive integer n. Invalid length: %s' % len(breakpoints))
        L = log2floor(len(breakpoints) - 1)
        S, B_LEFT, B_RIGHT = self._branching_scheme(L)
        polytopes = range(len(breakpoints) - 1)
        vertices = range(len(breakpoints))
        self.v = variable_dict()
        lmbda = self.v['lambda'] = variable_tuple((variable(lb=0) for v in vertices))
        y = self.v['y'] = variable_list((variable(domain_type=IntegerSet, lb=0, ub=1) for s in S))
        self.c = constraint_list()
        self.c.append(linear_constraint(variables=(self.input,) + tuple(lmbda), coefficients=(-1,) + breakpoints, rhs=0))
        self.c.append(linear_constraint(variables=(self.output,) + tuple(lmbda), coefficients=(-1,) + values))
        if self.bound == 'ub':
            self.c[-1].lb = 0
        elif self.bound == 'lb':
            self.c[-1].ub = 0
        else:
            assert self.bound == 'eq'
            self.c[-1].rhs = 0
        self.c.append(linear_constraint(variables=tuple(lmbda), coefficients=(1,) * len(lmbda), rhs=1))
        clist = []
        for s in S:
            variables = tuple((lmbda[v] for v in B_LEFT[s]))
            clist.append(linear_constraint(variables=variables + (y[s],), coefficients=(1,) * len(variables) + (-1,), ub=0))
        self.c.append(constraint_tuple(clist))
        del clist
        clist = []
        for s in S:
            variables = tuple((lmbda[v] for v in B_RIGHT[s]))
            clist.append(linear_constraint(variables=variables + (y[s],), coefficients=(1,) * len(variables) + (1,), ub=1))
        self.c.append(constraint_tuple(clist))

    def _branching_scheme(self, n):
        N = 2 ** n
        S = range(n)
        G = generate_gray_code(n)
        L = tuple(([k for k in range(N + 1) if (k == 0 or G[k - 1][s] == 1) and (k == N or G[k][s] == 1)] for s in S))
        R = tuple(([k for k in range(N + 1) if (k == 0 or G[k - 1][s] == 0) and (k == N or G[k][s] == 0)] for s in S))
        return (S, L, R)

    def validate(self, **kwds):
        """
        Validate this piecewise linear function by verifying
        various properties of the breakpoints, values, and
        input variable (e.g., that the list of breakpoints
        is nondecreasing).

        See base class documentation for keyword
        descriptions.
        """
        return super(piecewise_log, self).validate(**kwds)
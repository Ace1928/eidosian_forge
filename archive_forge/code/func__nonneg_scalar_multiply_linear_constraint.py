from pyomo.core import (
from pyomo.core.base import TransformationFactory, _VarData
from pyomo.core.plugins.transform.hierarchy import Transformation
from pyomo.common.config import ConfigBlock, ConfigValue, NonNegativeFloat
from pyomo.common.modeling import unique_component_name
from pyomo.repn.standard_repn import generate_standard_repn
from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.opt import TerminationCondition
import logging
def _nonneg_scalar_multiply_linear_constraint(self, cons, scalar):
    """Multiplies all coefficients and the RHS of a >= constraint by scalar.
        There is no logic for flipping the equality, so this is just the
        special case with a nonnegative scalar, which is all we need.

        If self.do_integer_arithmetic is True, this assumes that scalar is an
        int. It also will throw an error if any data is non-integer (within
        tolerance)
        """
    body = cons['body']
    new_coefs = []
    for i, coef in enumerate(body.linear_coefs):
        v = body.linear_vars[i]
        new_coefs.append(self._multiply(scalar, coef, self._get_noninteger_coef_error_message, (v.name, coef)))
        cons['map'][v] = new_coefs[i]
    body.linear_coefs = new_coefs
    body.quadratic_coefs = [scalar * coef for coef in body.quadratic_coefs]
    body.nonlinear_expr = scalar * body.nonlinear_expr if body.nonlinear_expr is not None else None
    lb = cons['lower']
    if lb is not None:
        cons['lower'] = self._multiply(scalar, lb, self._nonneg_scalar_multiply_linear_constraint_error_msg, (cons, coef))
    return cons
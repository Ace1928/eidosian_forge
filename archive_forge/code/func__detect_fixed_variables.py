from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.core.base.constraint import Constraint
from pyomo.core.base.transformation import TransformationFactory
from pyomo.core.base.suffix import Suffix
from pyomo.core.expr.numvalue import value
from pyomo.core.plugins.transform.hierarchy import IsomorphicTransformation
from pyomo.repn.standard_repn import generate_standard_repn
from pyomo.common.config import (
from pyomo.common.errors import InfeasibleConstraintException
def _detect_fixed_variables(m):
    """Detect fixed variables due to constraints of form var = const."""
    new_fixed_vars = ComponentSet()
    for constr in m.component_data_objects(ctype=Constraint, active=True, descend_into=True):
        if constr.equality and constr.body.polynomial_degree() == 1:
            repn = generate_standard_repn(constr.body)
            if len(repn.linear_vars) == 1 and repn.linear_coefs[0]:
                var = repn.linear_vars[0]
                coef = float(repn.linear_coefs[0])
                const = repn.constant
                var_val = (value(constr.lower) - value(const)) / coef
                var.fix(var_val)
                new_fixed_vars.add(var)
    return new_fixed_vars
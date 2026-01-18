from pyomo.core import (
from pyomo.core.base import TransformationFactory, _VarData
from pyomo.core.plugins.transform.hierarchy import Transformation
from pyomo.common.config import ConfigBlock, ConfigValue, NonNegativeFloat
from pyomo.common.modeling import unique_component_name
from pyomo.repn.standard_repn import generate_standard_repn
from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.opt import TerminationCondition
import logging
def _process_constraint(self, constraint):
    """Transforms a pyomo Constraint object into a list of dictionaries
        representing only >= constraints. That is, if the constraint has both an
        ub and a lb, it is transformed into two constraints. Otherwise it is
        flipped if it is <=. Each dictionary contains the keys 'lower',
        and 'body' where, after the process, 'lower' will be a constant, and
        'body' will be the standard repn of the body. (The constant will be
        moved to the RHS and we know that the upper bound is None after this).
        """
    body = constraint.body
    std_repn = generate_standard_repn(body)
    cons_dict = {'lower': value(constraint.lower), 'body': std_repn}
    upper = value(constraint.upper)
    constraints_to_add = [cons_dict]
    if upper is not None:
        if cons_dict['lower'] is not None:
            leq_side = {'lower': -upper, 'body': generate_standard_repn(-1.0 * body)}
            self._move_constant_and_add_map(leq_side)
            constraints_to_add.append(leq_side)
        else:
            cons_dict['lower'] = -upper
            cons_dict['body'] = generate_standard_repn(-1.0 * body)
    self._move_constant_and_add_map(cons_dict)
    return constraints_to_add
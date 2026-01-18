from pyomo.core import (
from pyomo.core.base import TransformationFactory, _VarData
from pyomo.core.plugins.transform.hierarchy import Transformation
from pyomo.common.config import ConfigBlock, ConfigValue, NonNegativeFloat
from pyomo.common.modeling import unique_component_name
from pyomo.repn.standard_repn import generate_standard_repn
from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.opt import TerminationCondition
import logging
def _add_linear_constraints_error_msg(self, cons1, cons2):
    return 'The do_integer_arithmetic flag was set to True, but while adding %s and %s, encountered a coefficient that is non-integer within the specified tolerance\nPlease set do_integer_arithmetic=False, increase integer_tolerance, or make your data integer.' % (cons1['body'].to_expression() >= cons1['lower'], cons2['body'].to_expression() >= cons2['lower'])
from pyomo.common.config import ConfigDict, ConfigValue
from pyomo.common.errors import DeveloperError
from pyomo.common.modeling import unique_component_name
from pyomo.contrib.piecewise import PiecewiseLinearFunction
from pyomo.contrib.piecewise.transform.piecewise_to_mip_visitor import (
from pyomo.core import (
from pyomo.core.base import Transformation
from pyomo.core.base.block import _BlockData, Block
from pyomo.core.util import target_list
from pyomo.gdp import Disjunct, Disjunction
from pyomo.gdp.util import is_child_of
from pyomo.network import Port
def _transform_constraint(self, constraint, descend_into_expressions):
    if not descend_into_expressions:
        return
    transBlock = self._get_transformation_block(constraint.parent_block())
    visitor = PiecewiseLinearToMIP(self._transform_pw_linear_expr, transBlock)
    _constraints = constraint.values() if constraint.is_indexed() else (constraint,)
    for c in _constraints:
        visitor.walk_expression((c.expr, c, 0))
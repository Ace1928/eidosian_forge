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
def _apply_to_impl(self, instance, **kwds):
    config = self.CONFIG(kwds.pop('options', {}))
    config.set_value(kwds)
    targets = config.targets
    if targets is None:
        targets = (instance,)
    knownBlocks = {}
    not_walking_exprs_msg = "When not descending into expressions, Constraints and Objectives are not valid targets. Please specify PiecewiseLinearFunction component and the Blocks containing them, or (at the cost of some performance in this transformation), set the 'descend_into_expressions' option to 'True'."
    for t in targets:
        if not is_child_of(parent=instance, child=t, knownBlocks=knownBlocks):
            raise ValueError("Target '%s' is not a component on instance '%s'!" % (t.name, instance.name))
        if t.ctype is PiecewiseLinearFunction:
            if config.descend_into_expressions:
                raise ValueError('When descending into expressions, the transformation cannot take PiecewiseLinearFunction components as targets. Please instead specify the Blocks, Constraints, and Objectives where your PiecewiseLinearFunctions have been used in expressions.')
            self._transform_piecewise_linear_function(t, config.descend_into_expressions)
        elif t.ctype is Block or isinstance(t, _BlockData):
            self._transform_block(t, config.descend_into_expressions)
        elif t.ctype is Constraint:
            if not config.descend_into_expressions:
                raise ValueError("Encountered Constraint target '%s':\n%s" % (t.name, not_walking_exprs_msg))
            self._transform_constraint(t, config.descend_into_expressions)
        elif t.ctype is Objective:
            if not config.descend_into_expressions:
                raise ValueError("Encountered Objective target '%s':\n%s" % (t.name, not_walking_exprs_msg))
            self._transform_objective(t, config.descend_into_expressions)
        else:
            raise ValueError("Target '%s' is not a PiecewiseLinearFunction, Block or Constraint. It was of type '%s' and can't be transformed." % (t.name, type(t)))
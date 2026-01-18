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
def _transform_block(self, block, descend_into_expressions):
    blocks = block.values() if block.is_indexed() else (block,)
    for b in blocks:
        for obj in b.component_objects(active=True, descend_into=(Block, Disjunct), sort=SortComponents.deterministic):
            handler = self.handlers.get(obj.ctype, None)
            if not handler:
                if handler is None:
                    raise RuntimeError("No transformation handler registered for modeling components of type '%s'." % obj.ctype)
                continue
            handler(obj, descend_into_expressions)
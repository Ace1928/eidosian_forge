from functools import wraps
from pyomo.common.collections import ComponentMap
from pyomo.common.log import is_debug_set
from pyomo.common.modeling import unique_component_name
from pyomo.core.base import Transformation, TransformationFactory
from pyomo.core.base.external import ExternalFunction
from pyomo.core import (
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.gdp.transformed_disjunct import _TransformedDisjunct
from pyomo.gdp.util import (
from pyomo.network import Port
from weakref import ref as weakref_ref
def _transform_block_components(self, block, disjunct, *args):
    varRefBlock = disjunct._transformation_block().localVarReferences
    for v in block.component_objects(Var, descend_into=Block, active=None):
        varRefBlock.add_component(unique_component_name(varRefBlock, v.getname(fully_qualified=False)), Reference(v))
    for obj in block.component_objects(active=True, descend_into=Block):
        handler = self.handlers.get(obj.ctype, None)
        if not handler:
            if handler is None:
                raise GDP_Error('No %s transformation handler registered for modeling components of type %s. If your disjuncts contain non-GDP Pyomo components that require transformation, please transform them first.' % (self.transformation_name, obj.ctype))
            continue
        handler(obj, disjunct, *args)
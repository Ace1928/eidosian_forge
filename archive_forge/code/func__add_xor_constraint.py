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
def _add_xor_constraint(self, disjunction, transBlock):
    if disjunction in self._algebraic_constraints:
        return self._algebraic_constraints[disjunction]
    if disjunction.is_indexed():
        orC = Constraint(Any)
    else:
        orC = Constraint()
    orCname = unique_component_name(transBlock, disjunction.getname(fully_qualified=False) + '_xor')
    transBlock.add_component(orCname, orC)
    self._algebraic_constraints[disjunction] = orC
    return orC
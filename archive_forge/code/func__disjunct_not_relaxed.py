import logging
from pyomo.core.base import Transformation, Block, Constraint
from pyomo.gdp import Disjunct, GDP_Error, Disjunction
from pyomo.core import TraversalStrategy, TransformationFactory
from pyomo.core.base.indexed_component import ActiveIndexedComponent
from pyomo.common.deprecation import deprecated
def _disjunct_not_relaxed(self, disjunct):
    return disjunct.transformation_block is None
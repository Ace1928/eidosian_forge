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
def _restore_state(self):
    self._transformation_blocks.clear()
    self._algebraic_constraints.clear()
    if hasattr(self, '_config'):
        del self._config
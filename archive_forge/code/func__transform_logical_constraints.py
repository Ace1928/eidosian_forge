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
def _transform_logical_constraints(self, instance, targets):
    disj_targets = []
    for t in targets:
        disj_datas = t.values() if t.is_indexed() else [t]
        if t.ctype is Disjunct:
            disj_targets.extend(disj_datas)
        if t.ctype is Disjunction:
            disj_targets.extend([d for disjunction in disj_datas for d in disjunction.disjuncts])
    TransformationFactory('contrib.logical_to_disjunctive').apply_to(instance, targets=[blk for blk in targets if blk.ctype is Block] + disj_targets)
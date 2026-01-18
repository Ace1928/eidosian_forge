import logging
from collections import defaultdict
from pyomo.common.autoslots import AutoSlots
import pyomo.common.config as cfg
from pyomo.common import deprecated
from pyomo.common.collections import ComponentMap, ComponentSet, DefaultComponentMap
from pyomo.common.modeling import unique_component_name
from pyomo.core.expr.numvalue import ZeroConstant
import pyomo.core.expr as EXPR
from pyomo.core.base import TransformationFactory, Reference
from pyomo.core import (
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.gdp.disjunct import _DisjunctData
from pyomo.gdp.plugins.gdp_to_mip_transformation import GDP_to_MIP_Transformation
from pyomo.gdp.transformed_disjunct import _TransformedDisjunct
from pyomo.gdp.util import (
from pyomo.core.util import target_list
from pyomo.util.vars_from_expressions import get_vars_from_components
from weakref import ref as weakref_ref
def _get_user_defined_local_vars(self, targets):
    user_defined_local_vars = defaultdict(ComponentSet)
    seen_blocks = set()
    for t in targets:
        if t.ctype is Disjunct:
            for b in t.component_data_objects(Block, descend_into=Block, active=True, sort=SortComponents.deterministic):
                if b not in seen_blocks:
                    self._collect_local_vars_from_block(b, user_defined_local_vars)
                    seen_blocks.add(b)
            blk = t
            while blk is not None:
                if blk in seen_blocks:
                    break
                self._collect_local_vars_from_block(blk, user_defined_local_vars)
                seen_blocks.add(blk)
                blk = blk.parent_block()
    return user_defined_local_vars
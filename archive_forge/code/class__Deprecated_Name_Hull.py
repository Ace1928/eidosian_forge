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
@TransformationFactory.register('gdp.chull', doc="[DEPRECATED] please use 'gdp.hull' to get the Hull transformation.")
@deprecated("The 'gdp.chull' name is deprecated. Please use the more apt 'gdp.hull' instead.", logger='pyomo.gdp', version='5.7')
class _Deprecated_Name_Hull(Hull_Reformulation):

    def __init__(self):
        super(_Deprecated_Name_Hull, self).__init__()
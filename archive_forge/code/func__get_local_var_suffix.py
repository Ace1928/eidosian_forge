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
def _get_local_var_suffix(self, disjunct):
    localSuffix = disjunct.component('LocalVars')
    if localSuffix is None:
        disjunct.LocalVars = Suffix(direction=Suffix.LOCAL)
    else:
        if localSuffix.ctype is Suffix:
            return
        raise GDP_Error("A component called 'LocalVars' is declared on Disjunct %s, but it is of type %s, not Suffix." % (disjunct.getname(fully_qualified=True), localSuffix.ctype))
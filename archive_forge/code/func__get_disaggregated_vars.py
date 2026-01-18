from pyomo.common.config import (
from pyomo.common.modeling import unique_component_name
from pyomo.core import (
from pyomo.core.expr import differentiate
from pyomo.common.collections import ComponentSet
from pyomo.opt import SolverFactory
from pyomo.repn import generate_standard_repn
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.gdp.util import (
from pyomo.contrib.fme.fourier_motzkin_elimination import (
import logging
def _get_disaggregated_vars(self, hull):
    disaggregatedVars = ComponentSet()
    for disjunction in hull.component_data_objects(Disjunction, descend_into=(Disjunct, Block)):
        for disjunct in disjunction.disjuncts:
            transBlock = disjunct.transformation_block
            if transBlock is not None:
                for v in transBlock.disaggregatedVars.component_data_objects(Var):
                    disaggregatedVars.add(v)
    return disaggregatedVars
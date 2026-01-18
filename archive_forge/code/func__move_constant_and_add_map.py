from pyomo.core import (
from pyomo.core.base import TransformationFactory, _VarData
from pyomo.core.plugins.transform.hierarchy import Transformation
from pyomo.common.config import ConfigBlock, ConfigValue, NonNegativeFloat
from pyomo.common.modeling import unique_component_name
from pyomo.repn.standard_repn import generate_standard_repn
from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.opt import TerminationCondition
import logging
def _move_constant_and_add_map(self, cons_dict):
    """Takes constraint in dictionary form already in >= form,
        and moves the constant to the RHS
        """
    body = cons_dict['body']
    constant = value(body.constant)
    cons_dict['lower'] -= constant
    body.constant = 0
    cons_dict['map'] = ComponentMap(zip(body.linear_vars, [value(coef) for coef in body.linear_coefs]))
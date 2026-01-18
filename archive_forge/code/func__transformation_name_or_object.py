import logging
from math import fabs
from pyomo.common.config import ConfigDict, ConfigValue, NonNegativeFloat
from pyomo.contrib.cp.transform.logical_to_disjunctive_program import (
from pyomo.core.base import Transformation, TransformationFactory
from pyomo.core.base.block import Block
from pyomo.core.expr.numvalue import value
from pyomo.gdp import GDP_Error
from pyomo.gdp.disjunct import Disjunct, Disjunction
from pyomo.gdp.plugins.bigm import BigM_Transformation
def _transformation_name_or_object(transformation_name_or_object):
    if isinstance(transformation_name_or_object, Transformation):
        return transformation_name_or_object
    xform = TransformationFactory(transformation_name_or_object)
    if xform is None:
        raise ValueError('Expected valid name for a registered Pyomo transformation. \n\tRecieved: %s' % transformation_name_or_object)
    return xform
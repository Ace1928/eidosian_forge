import logging
from pyomo.common import deprecated
from pyomo.core.base import Transformation, TransformationFactory, Var, Suffix, Reals

    This plugin relaxes integrality in a Pyomo model.
    
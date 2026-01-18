from pyomo.common.deprecation import RenamedClass
from pyomo.core.base.component import ModelComponentFactory
from pyomo.core.base.indexed_component import rule_wrapper
from pyomo.core.base.expression import (
from pyomo.dae.contset import ContinuousSet
from pyomo.dae.diffvar import DAE_Error
class SimpleIntegral(metaclass=RenamedClass):
    __renamed__new_class__ = ScalarIntegral
    __renamed__version__ = '6.0'
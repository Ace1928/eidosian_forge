import inspect
from pyomo.common.deprecation import deprecation_warning
from pyomo.core.base.component import ModelComponentFactory
from pyomo.core.base.transformation import (
from pyomo.scripting.interface import (
class IParamRepresentation(DeprecatedInterface):
    pass
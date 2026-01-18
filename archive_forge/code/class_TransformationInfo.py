from pyomo.common import Factory
from pyomo.common.collections import ComponentSet
from pyomo.common.errors import MouseTrap
from pyomo.common.deprecation import deprecated
from pyomo.common.modeling import unique_component_name
from pyomo.common.timing import TransformationTimer
class TransformationInfo(object):
    pass
import weakref
from pyomo.common.collections import ComponentMap
from pyomo.core.base.component import ModelComponentFactory
from pyomo.core.base.set import UnknownSetDimen
from pyomo.core.base.var import Var
from pyomo.dae.contset import ContinuousSet
class DAE_Error(Exception):
    """Exception raised while processing DAE Models"""
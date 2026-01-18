from pyomo.core.pyomoobject import PyomoObject
from pyomo.core.base.range import NonNumericRange
def _get_global_set(name):
    return GlobalSets[name]
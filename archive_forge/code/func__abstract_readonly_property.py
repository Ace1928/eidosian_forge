import copy
import weakref
from pyomo.common.autoslots import AutoSlots
def _abstract_readonly_property(**kwds):
    p = property(fget=_not_implemented, **kwds)
    return p
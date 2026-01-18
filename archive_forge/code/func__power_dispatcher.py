import numpy.core.numeric as nx
import numpy.core.numerictypes as nt
from numpy.core.numeric import asarray, any
from numpy.core.overrides import array_function_dispatch
from numpy.lib.type_check import isreal
def _power_dispatcher(x, p):
    return (x, p)
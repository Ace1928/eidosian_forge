from functools import reduce
from operator import mul
import sys
import warnings
from .util.arithmeticdict import ArithmeticDict
from .util.pyutil import NameSpace, deprecated
def _get_unit_from_registry(dimensionality, registry):
    return reduce(mul, [registry[k] ** v for k, v in dimensionality.items()])
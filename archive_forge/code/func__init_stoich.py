from collections import OrderedDict, defaultdict
from functools import reduce
from itertools import chain, product
from operator import mul, add
import copy
import math
import warnings
from .util.arithmeticdict import ArithmeticDict
from .util._expr import Expr
from .util.periodic import mass_from_composition
from .util.parsing import (
from .units import default_units, is_quantity, unit_of, to_unitless
from ._util import intdiv
from .util.pyutil import deprecated, DeferredImport, ChemPyDeprecationWarning
@staticmethod
def _init_stoich(container):
    if isinstance(container, set):
        container = {k: 1 for k in container}
    container = container or {}
    if type(container) == dict:
        container = OrderedDict(sorted(container.items(), key=lambda kv: kv[0]))
    return container
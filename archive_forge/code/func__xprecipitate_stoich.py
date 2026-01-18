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
def _xprecipitate_stoich(self, substances, xor):
    return tuple((0 if xor ^ (getattr(v, 'phase_idx', 0) > 0) else self.prod.get(k, 0) + self.inact_prod.get(k, 0) - self.reac.get(k, 0) - self.inact_reac.get(k, 0) for k, v in substances.items()))
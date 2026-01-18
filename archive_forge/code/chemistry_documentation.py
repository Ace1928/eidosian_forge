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
Multiplier of how many times rxn can be added/subtracted.

        Parameters
        ----------
        rxn : Equilibrium

        Examples
        --------
        >>> e1 = Equilibrium({'Cd(OH)2(s)': 4, 'H2O': 4},
        ...                  {'Cd4(OH)4+4': 1, 'H+': 4, 'OH-': 8}, 7.94e-91)
        >>> e2 = Equilibrium({'H2O': 1}, {'H+': 1, 'OH-': 1}, 10**-14)
        >>> e1.cancel(e2)
        -4
        >>> print(e1 - 4*e2)
        4 Cd(OH)2(s) = Cd4(OH)4+4 + 4 OH-; 7.94e-35

        
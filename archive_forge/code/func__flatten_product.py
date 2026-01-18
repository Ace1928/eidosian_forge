import inspect
import itertools
import logging
import math
import sys
import weakref
from pyomo.common.pyomo_typing import overload
from pyomo.common.collections import ComponentSet
from pyomo.common.deprecation import deprecated, deprecation_warning, RenamedClass
from pyomo.common.errors import DeveloperError, PyomoException
from pyomo.common.log import is_debug_set
from pyomo.common.modeling import NOTSET
from pyomo.common.sorting import sorted_robust
from pyomo.common.timing import ConstructionTimer
from pyomo.core.expr.numvalue import (
from pyomo.core.base.disable_methods import disable_methods
from pyomo.core.base.initializer import (
from pyomo.core.base.range import (
from pyomo.core.base.component import (
from pyomo.core.base.indexed_component import (
from pyomo.core.base.global_set import (
from collections.abc import Sequence
from operator import itemgetter
def _flatten_product(self, val):
    """Flatten any nested set product terms (due to nested products)

        Note that because this is called in a recursive context, this
        method is assured that there is no more than a single level of
        nested tuples (so this only needs to check the top-level terms)

        """
    for i in range(len(val) - 1, -1, -1):
        if val[i].__class__ is tuple:
            val = val[:i] + val[i] + val[i + 1:]
    return val
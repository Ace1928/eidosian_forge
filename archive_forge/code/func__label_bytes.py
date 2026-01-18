import time
import logging
import array
from weakref import ref as weakref_ref
from pyomo.common.log import is_debug_set
from pyomo.common.numeric_types import value
from pyomo.core.expr.numvalue import is_fixed, ZeroConstant
from pyomo.core.base.set_types import Any
from pyomo.core.base import SortComponents, Var
from pyomo.core.base.component import ModelComponentFactory
from pyomo.core.base.constraint import (
from pyomo.core.expr.numvalue import native_numeric_types
from pyomo.repn import generate_standard_repn
from collections.abc import Mapping
def _label_bytes(x):
    if x < 1000.0:
        return str(x) + ' B'
    if x < 1000000.0:
        return str(x / 1000.0) + ' KB'
    if x < 1000000000.0:
        return str(x / 1000000.0) + ' MB'
    return str(x / 1000000000.0) + ' GB'
import logging
import types
import weakref
from pyomo.common.pyomo_typing import overload
from ctypes import (
from pyomo.common.autoslots import AutoSlots
from pyomo.common.fileutils import find_library
from pyomo.core.expr.numvalue import (
import pyomo.core.expr as EXPR
from pyomo.core.base.component import Component
from pyomo.core.base.units_container import units
def _python_callback_fid_mapper(encode, val):
    if encode:
        return PythonCallbackFunction.global_registry[val]()
    else:
        return PythonCallbackFunction.register_instance(val)
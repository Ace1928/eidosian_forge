import logging
import sys
from weakref import ref as weakref_ref
import gc
import math
from pyomo.common import timing
from pyomo.common.collections import Bunch
from pyomo.common.dependencies import pympler, pympler_available
from pyomo.common.deprecation import deprecated
from pyomo.common.gc_manager import PauseGC
from pyomo.common.log import is_debug_set
from pyomo.common.numeric_types import value
from pyomo.core.staleflag import StaleFlagManager
from pyomo.core.expr.symbol_map import SymbolMap
from pyomo.core.base.component import ModelComponentFactory
from pyomo.core.base.var import Var
from pyomo.core.base.constraint import Constraint
from pyomo.core.base.objective import Objective
from pyomo.core.base.suffix import active_import_suffix_generator
from pyomo.core.base.block import ScalarBlock
from pyomo.core.base.set import Set
from pyomo.core.base.componentuid import ComponentUID
from pyomo.core.base.label import CNameLabeler, CuidLabeler
from pyomo.dataportal.DataPortal import DataPortal
from pyomo.opt.results import Solution, SolverStatus, UndefinedData
from contextlib import nullcontext
from io import StringIO
def _initialize_component(self, modeldata, namespaces, component_name, profile_memory):
    declaration = self.component(component_name)
    if component_name in modeldata._default:
        if declaration.ctype is not Set:
            declaration.set_default(modeldata._default[component_name])
    data = None
    for namespace in namespaces:
        if component_name in modeldata._data.get(namespace, {}):
            data = modeldata._data[namespace][component_name]
        if data is not None:
            break
    generate_debug_messages = is_debug_set(logger)
    if generate_debug_messages:
        _blockName = 'Model' if self.parent_block() is None else "Block '%s'" % self.name
        logger.debug("Constructing %s '%s' on %s from data=%s", declaration.__class__.__name__, declaration.name, _blockName, str(data))
    try:
        declaration.construct(data)
    except:
        err = sys.exc_info()[1]
        logger.error("Constructing component '%s' from data=%s failed:\n    %s: %s", str(declaration.name), str(data).strip(), type(err).__name__, err, extra={'cleandoc': False})
        raise
    if generate_debug_messages:
        _out = StringIO()
        declaration.pprint(ostream=_out)
        logger.debug("Constructed component '%s':\n    %s" % (declaration.name, _out.getvalue()))
    if profile_memory >= 2 and pympler_available:
        mem_used = pympler.muppy.get_size(pympler.muppy.get_objects())
        print('      Total memory = %d bytes following construction of component=%s' % (mem_used, component_name))
        if profile_memory >= 3:
            gc.collect()
            mem_used = pympler.muppy.get_size(pympler.muppy.get_objects())
            print('      Total memory = %d bytes following construction of component=%s (after garbage collection)' % (mem_used, component_name))
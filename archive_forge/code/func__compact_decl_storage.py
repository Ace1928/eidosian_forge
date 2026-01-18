import copy
import logging
import sys
import weakref
import textwrap
from collections import defaultdict
from contextlib import contextmanager
from inspect import isclass, currentframe
from io import StringIO
from itertools import filterfalse, chain
from operator import itemgetter, attrgetter
from pyomo.common.autoslots import AutoSlots
from pyomo.common.collections import Mapping
from pyomo.common.deprecation import deprecated, deprecation_warning, RenamedClass
from pyomo.common.formatting import StreamIndenter
from pyomo.common.gc_manager import PauseGC
from pyomo.common.log import is_debug_set
from pyomo.common.pyomo_typing import overload
from pyomo.common.timing import ConstructionTimer
from pyomo.core.base.component import (
from pyomo.core.base.enums import SortComponents, TraversalStrategy
from pyomo.core.base.global_set import UnindexedComponent_index
from pyomo.core.base.componentuid import ComponentUID
from pyomo.core.base.set import Any
from pyomo.core.base.var import Var
from pyomo.core.base.initializer import Initializer
from pyomo.core.base.indexed_component import (
from pyomo.opt.base import ProblemFormat, guess_format
from pyomo.opt import WriterFactory
def _compact_decl_storage(self):
    idxMap = {}
    _new_decl_order = []
    j = 0
    for i, entry in enumerate(self._decl_order):
        if entry[0] is not None:
            idxMap[i] = j
            j += 1
            _new_decl_order.append(entry)
    self._decl = {k: idxMap[idx] for k, idx in self._decl.items()}
    for ctype, info in self._ctypes.items():
        idx = info[0]
        entry = self._decl_order[idx]
        while entry[0] is None:
            idx = entry[1]
            entry = self._decl_order[idx]
        info[0] = last = idxMap[idx]
        while entry[1] is not None:
            idx = entry[1]
            entry = self._decl_order[idx]
            if entry[0] is not None:
                this = idxMap[idx]
                _new_decl_order[last] = (_new_decl_order[last][0], this)
                last = this
        info[1] = last
        _new_decl_order[last] = (_new_decl_order[last][0], None)
    self._decl_order = _new_decl_order
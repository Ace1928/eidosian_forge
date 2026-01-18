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
def _component_typemap(self, ctype=None, active=None, sort=False):
    """
        Return information about the block components.

        If ctype is None, return a dictionary that maps
           {component type -> {name -> component instance}}
        Otherwise, return a dictionary that maps
           {name -> component instance}
        for the specified component type.

        Note: The actual {name->instance} object is a PseudoMap that
        implements a lightweight interface to the underlying
        BlockComponents data structures.
        """
    if ctype is None:
        ans = {}
        for x in self._ctypes:
            ans[x] = PseudoMap(self, x, active, sort)
        return ans
    else:
        return PseudoMap(self, ctype, active, sort)
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
class _generic_component_decorator(object):
    """A generic decorator that wraps Block.__setattr__()

    Arguments
    ---------
        component: the Pyomo Component class to construct
        block: the block onto which to add the new component
        *args: positional arguments to the Component constructor
               (*excluding* the block argument)
        **kwds: keyword arguments to the Component constructor
    """

    def __init__(self, component, block, *args, **kwds):
        self._component = component
        self._block = block
        self._args = args
        self._kwds = kwds

    def __call__(self, rule):
        setattr(self._block, rule.__name__, self._component(*self._args, rule=rule, **self._kwds))
        return rule